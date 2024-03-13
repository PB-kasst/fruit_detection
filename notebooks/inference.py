from ultralytics import YOLO
import cv2
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    model = YOLO("C:/Users/eaguayo/Documents/ProjectBinder/General/Projects/00 DATASCIENCE/00_pilot/fruit_detection/notebooks/runs/detect/train3/weights/best.pt")

    # start webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                print(box)

                # # bounding box
                # x1, y1, x2, y2 = box.xyxy[0]
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # # put box in cam
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # # confidence
                # confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # # class name
                # cls = int(box.cls[0])
                # print("Class name -->", "Banana")

                # # object details
                # org = [x1, y1]
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale = 1
                # color = (255, 0, 0)
                # thickness = 2

                # cv2.putText(img, "Banana", org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()