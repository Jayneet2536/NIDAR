from ultralytics import YOLO
from download_model import weights_path

model = YOLO(weights_path)

results = model.predict(
    source="image.jpg",
    conf=0.25
)

results[0].show()