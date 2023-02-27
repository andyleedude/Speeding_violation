import cv2
import numpy as np
from deep_sort import DeepSort
from yolov5.detect import detect

import sys
sys.path.insert(0, './yolov5')

# Load YOLOv5 model
model = cv2.dnn.readNet('yolov5.weights', 'yolov5.cfg')

# Initialize DeepSORT tracker
deepsort = DeepSort()

# Open video file
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and classify vehicles using YOLOv5
    boxes, classes, scores = detect(frame, model)

    # Pass detections to DeepSORT
    features = deepsort.features(frame, boxes)
    trackers = deepsort.track(features, boxes, scores)

    # Draw bounding boxes and vehicle speeds on the frame
    for t in trackers:
        bbox = t.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(frame, 'Speed: {:.2f}'.format(t.speed), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Vehicle Speed Estimation', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
