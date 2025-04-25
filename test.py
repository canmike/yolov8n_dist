
import ultralytics
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import datetime

import ultralytics.utils.checks as checks
# Override the AMP‐allclose check so it never rebuilds the model
checks.check_amp = lambda model, *args, **kwargs: True

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = YOLO("yolov8n.yaml")
    # model.load("yolov8n.pt")
    # model.load("best_yolov8n_depth.pt")  # Load the model weights

    # model = YOLO("best_yolov8n_depth.pt").to(device)  # Load the model weights
    # model = YOLO("best_yolov8n_distance.pt")
    # model = YOLO("yolov8n.yaml") 
    # model.load("best_yolov8n_depth.pt")
    model = YOLO("best_yolov8n_distance_v2.pt")

    # model = YOLO("yolov8n.pt")  # or yolov8s.pt

    # path = r"C:\Users\can.michael\Downloads\car.jpg"
    path = r'C:\Users\can.michael\Desktop\is\SafezoneVision\dataset\dataset_pool\pg_1_0_1_subset\images\pg1_0023.jpg'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = (
        torch.from_numpy(img).unsqueeze(0).float().permute(0, 3, 1, 2)
    )  # Convert to CHW format
    x /= 255.0  # Normalize to [0, 1]
    x.shape
    # resize to (640, 640) if needed
    if x.shape[2] != 640 or x.shape[3] != 640:
        x = torch.nn.functional.interpolate(x, size=(640, 640), mode="bilinear")
    x.shape
    y = model(x)
    y[0].boxes
    y[0].dists

    model.train(
        data=r"C:\Users\can.michael\Desktop\is\SafezoneVision\dataset\dataset_pool\pg_1_0_1_subset\data.yaml",
        epochs=1000,
        imgsz=640,
        batch=4,  # or smaller if low VRAM
        workers=0,
        name=f"model_yolov8dist_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
        project="runs",  # 👈 optional: sets root directory
        cache=False,
        cls=3,
        amp=False,
        pretrained=True,
        device=device,
        # resume=True
    )

    # model.export(format="imx", int8=True, imgsz=[640, 640],
    #                  data="data.yaml", opset=11, name="yolo", project="yolo")


    next(model.parameters()).device
    # model.save("best_yolov8n_distance_v2.pt")  # Save the model weights

if __name__ == "__main__":
    main()