const VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
];

const COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"
];

let session;
const modelPath = 'models/best.onnx';

async function initModel() {
    const statusEl = document.getElementById('demo-status');
    try {
        statusEl.innerText = "Loading ONNX model (may take a moment)...";
        // Initialize ONNX Runtime session
        session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
        statusEl.innerText = "Model loaded successfully. Ready for inference.";
        document.getElementById('run-inference-btn').disabled = false;
    } catch (e) {
        statusEl.innerText = "Error loading model: " + e.message;
        console.error(e);
    }
}

document.getElementById('image-upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = document.getElementById('demo-image');
        img.src = event.target.result;
        img.onload = () => {
            // clear previous canvas
            const canvas = document.getElementById('demo-canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.getElementById('demo-status').innerText = "Image loaded. Click 'Run Inference'.";
        };
    };
    reader.readAsDataURL(file);
});

document.getElementById('run-inference-btn').addEventListener('click', async () => {
    const img = document.getElementById('demo-image');
    if (!img.src) return;
    
    const statusEl = document.getElementById('demo-status');
    statusEl.innerText = "Running inference...";
    
    // Disable button to prevent double-clicks
    document.getElementById('run-inference-btn').disabled = true;
    
    try {
        // give UI time to update
        await new Promise(r => setTimeout(r, 50));
        await runInference(img);
        statusEl.innerText = "Inference complete.";
    } catch (e) {
        statusEl.innerText = "Inference failed: " + e.message;
        console.error(e);
    } finally {
        document.getElementById('run-inference-btn').disabled = false;
    }
});

function iou(box1, box2) {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);
    
    const intersect = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    return intersect / (area1 + area2 - intersect);
}

function nms(boxes, iou_threshold) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];
    for (let i = 0; i < boxes.length; i++) {
        let keep = true;
        for (let j = 0; j < selected.length; j++) {
            if (boxes[i].class_id === selected[j].class_id && iou(boxes[i].box, selected[j].box) > iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push(boxes[i]);
        }
    }
    return selected;
}

async function runInference(imageSource) {
    const size = 640;
    
    // Create an offscreen canvas to scale the image
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Calculate letterbox scaling
    const scale = Math.min(size / imageSource.naturalWidth, size / imageSource.naturalHeight);
    const newW = imageSource.naturalWidth * scale;
    const newH = imageSource.naturalHeight * scale;
    const padX = (size - newW) / 2;
    const padY = (size - newH) / 2;
    
    ctx.fillStyle = '#727272'; // padding color
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(imageSource, padX, padY, newW, newH);
    
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const floatData = new Float32Array(3 * size * size);
    
    // Convert to NCHW float array
    for (let i = 0; i < size * size; i++) {
        floatData[i] = imgData[i * 4] / 255.0; // R
        floatData[size * size + i] = imgData[i * 4 + 1] / 255.0; // G
        floatData[2 * size * size + i] = imgData[i * 4 + 2] / 255.0; // B
    }
    
    const tensor = new ort.Tensor('float32', floatData, [1, 3, size, size]);
    
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const output = results[session.outputNames[0]].data;
    
    // YOLO11 output shape is typically [1, 4+num_classes, 8400]
    const num_classes = VISDRONE_CLASSES.length;
    const num_anchors = output.length / (4 + num_classes);
    
    const boxes = [];
    
    for (let i = 0; i < num_anchors; i++) {
        let max_score = 0;
        let class_id = -1;
        
        for (let c = 0; c < num_classes; c++) {
            const score = output[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score > 0.25) {
            const cx = output[0 * num_anchors + i];
            const cy = output[1 * num_anchors + i];
            const w  = output[2 * num_anchors + i];
            const h  = output[3 * num_anchors + i];
            
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;
            
            // Map back to original image space
            const orig_x1 = (x1 - padX) / scale;
            const orig_y1 = (y1 - padY) / scale;
            const orig_x2 = (x2 - padX) / scale;
            const orig_y2 = (y2 - padY) / scale;
            
            boxes.push({ box: [orig_x1, orig_y1, orig_x2, orig_y2], score: max_score, class_id: class_id });
        }
    }
    
    const final_boxes = nms(boxes, 0.45);
    drawBoxes(final_boxes, imageSource.naturalWidth, imageSource.naturalHeight);
}

function drawBoxes(boxes, imgW, imgH) {
    const canvas = document.getElementById('demo-canvas');
    const imgElement = document.getElementById('demo-image');
    
    // Set canvas dimensions to match the displayed image dimensions
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleX = canvas.width / imgW;
    const scaleY = canvas.height / imgH;
    
    boxes.forEach(b => {
        const [x1, y1, x2, y2] = b.box;
        const color = COLORS[b.class_id % COLORS.length];
        
        const cx1 = x1 * scaleX;
        const cy1 = y1 * scaleY;
        const cw = (x2 - x1) * scaleX;
        const ch = (y2 - y1) * scaleY;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(cx1, cy1, cw, ch);
        
        ctx.fillStyle = color;
        const text = `${VISDRONE_CLASSES[b.class_id]} ${(b.score * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(cx1, cy1 - 16, textWidth + 4, 16);
        
        ctx.fillStyle = "#000";
        ctx.font = "12px monospace";
        ctx.fillText(text, cx1 + 2, cy1 - 4);
    });
}

// Load model on startup
window.onload = initModel;
