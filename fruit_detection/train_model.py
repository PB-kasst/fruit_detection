DEPENDENCIES = False

if DEPENDENCIES:
    !pip install roboflow ultralytics

from roboflow import Roboflow
from ultralytics import YOLO
import os

def download_data():
    rf = Roboflow(api_key="lAoJR2oOr2ty3W0sb0qn")
    project = rf.workspace("00mvpbanana").project("banana-detetot")
    version = project.version(1)
    dataset = version.download("yolov8")


def main():
    model = YOLO('yolov8n.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='Banana-Detetot-1/data.yaml', epochs=150)

if __name__ == "__main__":
    download_data()
    main()