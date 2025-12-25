from ultralytics import YOLO

model = YOLO("model/best.pt")

# Webcam (0 = default camera)
model.predict(source=0, show=True, conf=0.4)
