
### For photo

from ultralytics import YOLO
import cv2
import argparse
import os
import sys

# parse command-line arguments so user can specify an image/video path
parser = argparse.ArgumentParser(description="Run PAN detection on an image or video")
parser.add_argument(
    "source",
    nargs="?",
    default=None,
    help="Path to image/video file (or leave empty to edit the script)",
)
args = parser.parse_args()

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# determine source
if args.source:
    image_path = args.source
else:
    # fallback to a sample image that exists in this repository
    # feel free to change this or always pass the path as an argument
    image_path = os.path.join("dataset", "images", "val", "pan8.jpg")

# verify file exists
if not os.path.exists(image_path):
    print(f"Error: source '{image_path}' does not exist.")
    sys.exit(1)

results = model(image_path)

# Show results (display or save fallback)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
for idx, result in enumerate(results):
    frame = result.plot()
    try:
        cv2.imshow("Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        # likely no GUI support; save the annotated frame instead
        out_path = os.path.join(output_dir, f"result_{idx}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved result image to {out_path} (imshow unavailable)")



### For video detection

# from ultralytics import YOLO
# import cv2

# model = YOLO("runs/detect/train/weights/best.pt")
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     results = model(frame)

#     annotated_frame = results[0].plot()

#     cv2.imshow("PAN Card Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()