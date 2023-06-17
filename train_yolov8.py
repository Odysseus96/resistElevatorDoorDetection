from ultralytics import YOLO

model = YOLO("/project/train/src_repo/ev_detection/yolov8s.pt")

model.train(data="/project/train/src_repo/ev_detection/data/EVDATA.yaml", project="/project/train/models", epochs=100, mixup=0.5, batch=32)
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format="onnx", dynamic=True)  # export the model to ONNX format