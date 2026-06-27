# Aerial Scene Understanding Using YOLO 🛸

An FYP project for detecting objects in aerial drone imagery using YOLOv8.

## Demo Flow

```
Upload aerial image → YOLO detects objects → Bounding boxes shown → Object count displayed → Output saved
```

## Project Structure

```
Autonom/
├── app.py              # Streamlit web app
├── inference.py        # YOLO inference logic
├── requirements.txt    # Python dependencies
├── README.md
├── models/
│   └── best.pt         # Trained YOLO model weights
├── sample_images/      # Test aerial images
├── outputs/            # Saved detection results
├── notebooks/
│   └── drone.ipynb     # Training & evaluation notebook
└── report/
    └── ML-Proposal.pdf # Project proposal
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Detectable Classes

| Class | Description |
|---|---|
| Car | Passenger vehicles |
| Truck | Heavy goods vehicles |
| Bus | Public transport |
| Pedestrian | People on foot |
| Bicycle | Cycles |
| Motorcycle | Two-wheelers |

## Presentation Outline

1. **Proposal** – Problem statement & objectives
2. **Dataset** – Aerial imagery dataset details
3. **Model Training** – YOLOv8 training pipeline
4. **Evaluation Results** – mAP, precision, recall curves
5. **Streamlit Demo** – Live inference walkthrough
6. **Limitations** – Current constraints
7. **Future Work** – Planned improvements (real-time video, tracking, etc.)
