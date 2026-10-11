from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="dronefreak/visdrone-yolo11n",
    filename="best.pt"
)

model = YOLO(weights)

results = model.predict(
    source="image.jpg",
    conf=0.25
)

results[0].show()