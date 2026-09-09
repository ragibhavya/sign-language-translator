import React, { useRef, useEffect, useState } from "react";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS } from "@mediapipe/hands";
import * as tf from "@tensorflow/tfjs";
const LABELS = ["hello", "goodbye", "yes", "no", "thankyou", "please", "love", "help", "good", "sorry", "stop", "one", "two", "okay", "iloveyou", "water", "eat", "drink", "name", "my"];
const CONFIDENCE_THRESHOLD = 0.75;
const STABILITY_FRAMES = 15;

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const latestLandmarks = useRef(null);
  const modelRef = useRef(null);
  const lastSpokenRef = useRef("");
  const recentGuessesRef = useRef([]);
  const [status, setStatus] = useState("Loading camera...");
  const [prediction, setPrediction] = useState("");
  const [handsDetected, setHandsDetected] = useState(0);
  const [dataset, setDataset] = useState([]);
  const [counts, setCounts] = useState({});
  const [sentence, setSentence] = useState([]);
  const [mode, setMode] = useState("record");

  const speak = (word) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  };

  const flattenLandmarks = (multiHandLandmarks) =>
    multiHandLandmarks.flatMap((hand) => hand.flatMap((p) => [p.x, p.y, p.z]));

  const padTo126 = (arr) => {
    const result = arr.slice(0, 126);
    while (result.length < 126) result.push(0);
    return result;
  };

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    hands.onResults((results) => {
      const canvasCtx = canvasRef.current.getContext("2d");
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
          drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#6fd28a", lineWidth: 2 });
          drawLandmarks(canvasCtx, landmarks, { color: "#f2b544", lineWidth: 1 });
        }
        latestLandmarks.current = results.multiHandLandmarks;
        setHandsDetected(results.multiHandLandmarks.length);

        if (mode === "predict" && modelRef.current) {
          const flat = flattenLandmarks(results.multiHandLandmarks);
          const padded = padTo126(flat);
          const inputTensor = tf.tensor2d([padded]);
          const output = modelRef.current.predict(inputTensor);
          const scores = output.dataSync();
          const maxIndex = scores.indexOf(Math.max(...scores));
          const confidence = scores[maxIndex];

          let currentGuess = confidence >= CONFIDENCE_THRESHOLD ? LABELS[maxIndex] : "none";

          recentGuessesRef.current.push(currentGuess);
          if (recentGuessesRef.current.length > STABILITY_FRAMES) recentGuessesRef.current.shift();

          const allSame = recentGuessesRef.current.every((g) => g === recentGuessesRef.current[0]);
          const stableGuess =
            allSame && recentGuessesRef.current.length === STABILITY_FRAMES
              ? recentGuessesRef.current[0]
              : null;

          if (stableGuess && stableGuess !== "none") {
            setPrediction(`${stableGuess} (${(confidence * 100).toFixed(0)}%)`);
            if (lastSpokenRef.current !== stableGuess) {
              speak(stableGuess);
              lastSpokenRef.current = stableGuess;
              setSentence((prev) => [...prev, stableGuess]);
            }
          } else if (stableGuess === "none") {
            setPrediction("Sign not recognized");
            lastSpokenRef.current = "";
          }
          inputTensor.dispose();
          output.dispose();
        }
      } else {
        latestLandmarks.current = null;
        setHandsDetected(0);
        lastSpokenRef.current = "";
        recentGuessesRef.current = [];
      }
      canvasCtx.restore();
    });

    const camera = new Camera(videoRef.current, {
      onFrame: async () => { await hands.send({ image: videoRef.current }); },
      width: 640,
      height: 480,
    });
    camera.start();
    setStatus("Camera ready.");
  }, [mode]);

  const recordSign = (label) => {
    if (!latestLandmarks.current) {
      alert("No hand detected!");
      return;
    }
    const flat = flattenLandmarks(latestLandmarks.current);
    setDataset((prev) => [...prev, { label, data: flat }]);
    setCounts((prev) => ({ ...prev, [label]: (prev[label] || 0) + 1 }));
  };

  const loadExistingDataset = async () => {
    const response = await fetch("/signspeak-dataset.json");
    const existing = await response.json();
    setDataset((prev) => [...prev, ...existing]);
    const newCounts = {};
    existing.forEach((item) => {
      newCounts[item.label] = (newCounts[item.label] || 0) + 1;
    });
    setCounts((prev) => ({ ...prev, ...newCounts }));
    setStatus(`Loaded ${existing.length} existing samples.`);
  };

  const downloadDataset = () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "signspeak-dataset.json";
    a.click();
  };

  const trainModel = async () => {
    setStatus("Loading dataset...");
    const response = await fetch("/signspeak-dataset.json");
    const loadedDataset = await response.json();

    setStatus(`Preparing ${loadedDataset.length} samples...`);
    const xs = loadedDataset.map((item) => padTo126(item.data));
    const ys = loadedDataset.map((item) => LABELS.indexOf(item.label));

    const xTensor = tf.tensor2d(xs);
    const yTensor = tf.oneHot(tf.tensor1d(ys, "int32"), LABELS.length);

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [126], units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: LABELS.length, activation: "softmax" }));
    model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

    setStatus("Training...");
    await model.fit(xTensor, yTensor, {
      epochs: 80,
      batchSize: 8,
      callbacks: {
        onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch + 1}: acc = ${(logs.acc * 100).toFixed(1)}%`),
      },
    });

    modelRef.current = model;
    setStatus("Training complete!");
    setMode("predict");
    xTensor.dispose();
    yTensor.dispose();
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h2>SignSpeak</h2>
      <p>{status}</p>
      <p style={{ fontWeight: "bold" }}>Hands detected: {handsDetected}</p>
      <video ref={videoRef} style={{ display: "none" }}></video>
      <canvas ref={canvasRef} width="640" height="480" style={{ border: "2px solid black" }}></canvas>

      <div style={{ marginTop: "16px" }}>
        <button onClick={() => setMode(mode === "record" ? "predict" : "record")}>
          Switch to {mode === "record" ? "Predict" : "Record"} Mode
        </button>
      </div>

      {mode === "record" && (
        <div style={{ marginTop: "16px" }}>
          <button onClick={loadExistingDataset}>Load Existing Dataset</button>
          <div style={{ marginTop: "10px" }}>
            {LABELS.map((label) => (
              <button key={label} onClick={() => recordSign(label)} style={{ margin: "4px" }}>
                Record "{label}" ({counts[label] || 0})
              </button>
            ))}
          </div>
          <div style={{ marginTop: "10px" }}>
            <button onClick={downloadDataset}>Download Dataset ({dataset.length} samples)</button>
          </div>
        </div>
      )}

      {mode === "predict" && (
        <div style={{ marginTop: "16px" }}>
          <button onClick={trainModel}>Train Model</button>
          {prediction && (
            <div style={{ marginTop: "16px", fontSize: "28px", fontWeight: "bold", color: prediction.includes("not recognized") ? "gray" : "green" }}>
              {prediction}
            </div>
          )}
          <div style={{ marginTop: "16px" }}>
            <p style={{ fontWeight: "bold" }}>Sentence:</p>
            <p style={{ fontSize: "20px" }}>{sentence.join(" — ")}</p>
            <button onClick={() => setSentence([])}>Clear Sentence</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;