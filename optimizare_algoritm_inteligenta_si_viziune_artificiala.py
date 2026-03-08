from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="torchscript")
model.export(format = "onnx", imgsz = 640, opset = 14)

model_seg = YOLO("yolov8s-seg.pt")
model_seg.export(format = "onnx", opset = 19, dynamic = False)



