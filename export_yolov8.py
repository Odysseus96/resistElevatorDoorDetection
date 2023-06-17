from ultralytics import YOLO

model = YOLO("models/elevator-close-detv8s.pt")
success = model.export(format="onnx")  # export the model to ONNX format