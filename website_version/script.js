const CANVAS_SIZE = 560;
const PREVIEW_SIZE = 140;
const CANVAS_SCALE = 1;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");
const previewCanvas = document.getElementById("preview");
const previewCtx = previewCanvas.getContext("2d");

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./onnx_model.onnx");

ctx.lineWidth = 56;
ctx.lineJoin = "round";
ctx.font = "56px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "black";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

ctx.strokeStyle = "black";

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  previewCtx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col";
    element.children[0].children[0].style.height = "0";
  }
}

function drawLine(fromX, fromY, toX, toY) {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((value) => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map((value) => value / sum);
}

async function updatePredictions() {
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  console.log(imgData);

  const floatArray = new Float32Array(28 * 28);
  const previewImageData = previewCtx.createImageData(28, 28);

  const mean = 0.1307;
  const std = 0.3081;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let total = 0;
      for (let yy = 0; yy < 20; yy++) {
        for (let xx = 0; xx < 20; xx++) {
          const index = 4 * ((y * 20 + yy) * CANVAS_SIZE + (x * 20 + xx));
          const r = imgData.data[index + 3]; // Using the alpha channel for grayscale value
          total += r;
        }
      }
      const gray = total / (20 * 20 * 255); // Average and normalize to 0-1

      floatArray[y * 28 + x] = (gray - mean) / std; // Normalize using mean and std

      // Set preview image data for debugging
      const previewIndex = 4 * (y * 28 + x);
      previewImageData.data[previewIndex] = gray * 255;
      previewImageData.data[previewIndex + 1] = gray * 255;
      previewImageData.data[previewIndex + 2] = gray * 255;
      previewImageData.data[previewIndex + 3] = 255;
    }
  }

  const scaledPreviewImageData = previewCtx.createImageData(
    PREVIEW_SIZE,
    PREVIEW_SIZE
  );
  for (let y = 0; y < PREVIEW_SIZE; y++) {
    for (let x = 0; x < PREVIEW_SIZE; x++) {
      const srcX = Math.floor(x / 5);
      const srcY = Math.floor(y / 5);
      const srcIndex = (srcY * 28 + srcX) * 4;
      const destIndex = (y * PREVIEW_SIZE + x) * 4;
      scaledPreviewImageData.data[destIndex] = previewImageData.data[srcIndex];
      scaledPreviewImageData.data[destIndex + 1] =
        previewImageData.data[srcIndex + 1];
      scaledPreviewImageData.data[destIndex + 2] =
        previewImageData.data[srcIndex + 2];
      scaledPreviewImageData.data[destIndex + 3] =
        previewImageData.data[srcIndex + 3];
    }
  }

  previewCtx.putImageData(scaledPreviewImageData, 0, 0);

  console.log("Preview Input Data (28x28):", floatArray);

  const inputTensor = new onnx.Tensor(floatArray, "float32", [1, 1, 28, 28]);

  const outputMap = await sess.run([inputTensor]);
  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data;

  console.log("Model Output Data:", predictions);

  const probabilities = softmax(predictions);

  const maxPrediction = Math.max(...probabilities);
  for (let i = 0; i < probabilities.length; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.children[0].children[0].style.height = `${probabilities[i] * 100}%`;
    element.className =
      probabilities[i] === maxPrediction
        ? "prediction-col top-prediction"
        : "prediction-col";
  }
}

function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

loadingModelPromise.then(() => {
  canvas.addEventListener("mousedown", canvasMouseDown);
  canvas.addEventListener("mousemove", canvasMouseMove);
  document.body.addEventListener("mouseup", bodyMouseUp);
  document.body.addEventListener("mouseout", bodyMouseOut);
  clearButton.addEventListener("mousedown", clearCanvas);

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
});