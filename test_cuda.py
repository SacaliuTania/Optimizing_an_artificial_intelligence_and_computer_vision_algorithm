from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model("path_to_your_image.jpg")

# Show results
results.show()
