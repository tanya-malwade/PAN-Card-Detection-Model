import torch
import ultralytics.nn.tasks

torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
from ultralytics import YOLO

def main():
    # Load YOLO pretrained model
    model = YOLO("yolov8n.pt")

    # determine device: use GPU if available, otherwise fall back to CPU
    try:
        import torch
        if torch.cuda.is_available():
            device = 0  # first GPU
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    # Train the model
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=416,
        batch=8,
        device=device,
    )

if __name__ == "__main__":
    main()