#best-yolov10-60epochs-version4
# Import All the Required Libraries
# Import All the Required Libraries
import cv2
import math
import time
import torch
import os
from collections import defaultdict
from ultralytics import YOLOv10
import supervision as sv
from utils.object_tracking import ObjectTracking
import requests
import json

# Initialize ObjectTracking and DeepSORT
objectTracking = ObjectTracking()
deepsort = objectTracking.initialize_deepsort()

# Create a Video Capture Object for Webcam
cap = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 for external webcam

# Load Models
retrained_model = YOLOv10("weights/yolov10n.pt")  # For person detection
custom_model = YOLOv10("weights/best-yolov10-60epochs-version4.pt")  # For custom class detection

# Custom Class Names (Replace with your custom classes)
customClassNames = [
    "mask", "hairnet", "glove", "incorrect_mask", "no_mask", "no_hairnet", "no_glove"
]

# COCO Class Names (for person detection)
cocoClassNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Dictionary to store person violations
person_status = defaultdict(lambda: {
    "mask": False,
    "glove": False,
    "hairnet": False,
    "incorrect_mask": False,
    "no_mask": False,
    "no_hairnet": False,
    "no_glove": False
})

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# Function to map class IDs to class names
def get_class_name(class_id, model_type):
    if model_type == "retrained":
        return cocoClassNames[class_id] if class_id < len(cocoClassNames) else "unknown"
    elif model_type == "custom":
        return customClassNames[class_id] if class_id < len(customClassNames) else "unknown"
    return "unknown"

# Directory to save violation frames
save_dir = "violation_frames"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Initialize supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame processing loop
ctime = 0
ptime = 0
count = 0
while True:
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    ret, frame = cap.read()
    if ret:
        count += 1
        print(f"Frame Count: {count}")

        # Detect persons using the retrained model
        person_results = retrained_model.predict(frame, conf=0.25)
        print("Person Detections:")
        for box in person_results[0].boxes:
            print(f"Class: {get_class_name(int(box.cls[0]), 'retrained')}, Confidence: {box.conf[0]}")

        for result in person_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bbox_width = abs(x1 - x2)
                bbox_height = abs(y1 - y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                xywh_bboxs.append(xcycwh)
                conf = math.ceil(box.conf[0] * 100) / 100
                confs.append(conf)
                classNameInt = int(box.cls[0])
                oids.append(classNameInt)

        # Track people using DeepSORT
        xywhs = torch.tensor(xywh_bboxs)
        confidence = torch.tensor(confs)
        outputs = deepsort.update(xywhs, confidence, oids, frame)

        # Detect custom objects using the custom model
        custom_results = custom_model.predict(frame, conf=0.25)
        print("Custom Object Detections:")
        for box in custom_results[0].boxes:
            print(f"Class: {get_class_name(int(box.cls[0]), 'custom')}, Confidence: {box.conf[0]}")

        custom_bboxs = []
        custom_confs = []
        custom_oids = []
        for result in custom_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bbox_width = abs(x1 - x2)
                bbox_height = abs(y1 - y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                custom_bboxs.append(xcycwh)
                conf = math.ceil(box.conf[0] * 100) / 100
                custom_confs.append(conf)
                classNameInt = int(box.cls[0])
                custom_oids.append(classNameInt)

        # Annotate custom object detections using supervision
        custom_detections = sv.Detections.from_ultralytics(custom_results[0])
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=custom_detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=custom_detections)

        # Assign custom objects to tracked people
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            classIDs = outputs[:, -1]

            for i, bbox in enumerate(bbox_xyxy):
                person_id = identities[i]
                person_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # Convert to (x, y, w, h)

                # Reset status for each person
                status = {
                    "mask": False,
                    "glove": False,
                    "hairnet": False,
                    "incorrect_mask": False,
                    "no_mask": False,
                    "no_hairnet": False,
                    "no_glove": False
                }

                # Check for overlaps with custom objects
                for j, gear_box in enumerate(custom_bboxs):
                    obj_class = custom_oids[j]
                    class_name = get_class_name(obj_class, "custom")

                    # Convert gear box from center (cx, cy, w, h) to top-left (x1, y1, w, h)
                    gcx, gcy, gw, gh = gear_box
                    gx1, gy1 = int(gcx - gw // 2), int(gcy - gh // 2)
                    gear_box_xywh = [gx1, gy1, gw, gh]

                    px, py, pw, ph = person_bbox
                    is_inside = (px <= gcx <= px + pw) and (py <= gcy <= py + ph)
                    iou = calculate_iou(person_bbox, gear_box_xywh)

                    print(f"Person {person_id} BBOX: {person_bbox}")
                    print(f"{class_name} BBOX: {gear_box}, Center: ({gcx},{gcy})")
                    print(f"IoU: {iou:.3f}, Inside: {is_inside}")

                    if iou > 0.01 or is_inside:
                        print(f"[MATCH] person_id={person_id}, gear={class_name}, iou={iou:.3f}, inside={is_inside}")
                        
                        # Update status based on gear class
                        if class_name == "mask":
                            status["mask"] = True
                            status["no_mask"] = False
                            status["incorrect_mask"] = False
                        elif class_name == "glove":
                            status["glove"] = True
                            status["no_glove"] = False
                        elif class_name == "hairnet":
                            status["hairnet"] = True
                            status["no_hairnet"] = False
                        elif class_name == "incorrect_mask":
                            status["incorrect_mask"] = True
                            status["mask"] = False
                            status["no_mask"] = False
                        elif class_name == "no_mask":
                            status["no_mask"] = True
                            status["mask"] = False
                            status["incorrect_mask"] = False
                        elif class_name == "no_hairnet":
                            status["no_hairnet"] = True
                            status["hairnet"] = False
                        elif class_name == "no_glove":
                            status["no_glove"] = True
                            status["glove"] = False

                # Save updated status
                person_status[person_id] = status

                # Check for violations and save frames
                violations = [
                    status["no_mask"],
                    status["no_glove"],
                    status["no_hairnet"],
                    status["incorrect_mask"]
                ]
                
                print(f"Person ID: {person_id}, Status: {status}")
                print(f"Violations for Person ID: {person_id}: {violations}")

                if any(violations):
                    print(f"Violation detected for Person ID: {person_id} - Saving frame.")
                    violation_text = f"Person ID: {person_id} - " \
                                     f"Mask: {status['mask']}, " \
                                     f"Glove: {status['glove']}, " \
                                     f"Hairnet: {status['hairnet']}, " \
                                     f"Incorrect Mask: {status['incorrect_mask']}, " \
                                     f"No Mask: {status['no_mask']}, " \
                                     f"No Hairnet: {status['no_hairnet']}, " \
                                     f"No Glove: {status['no_glove']}"
                    cv2.putText(annotated_frame, violation_text, (10, 150 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Save the frame
                    cv2.imwrite(f"{save_dir}/violation_frame_{count}_person_{person_id}.jpg", annotated_frame)

                # Send to Flask server
                converted_status = {str(int(k)): v for k, v in person_status.items()}
                requests.post("http://127.0.0.1:5000/update", json=converted_status)   

        # Display FPS and frame count
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(annotated_frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(annotated_frame, f"Frame Count: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the annotated frame
        cv2.imshow("Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()












































# #Import All the Required Libraries
# import cv2
# import math
# import time
# import torch
# from ultralytics import YOLOv10
# from utils.object_tracking import ObjectTracking
# objectTracking = ObjectTracking()
# deepsort = objectTracking.initialize_deepsort()

# #Create a Video Capture Object
# cap = cv2.VideoCapture("Resources/video3.mp4")
# model = YOLOv10("weights/yolov10n.pt")
# cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]
# ctime = 0
# ptime = 0
# count = 0
# while True:
#     xywh_bboxs = []
#     confs = []
#     oids = []
#     outputs = []
#     ret, frame = cap.read()
#     if ret:
#         count += 1
#         print(f"Frame Count: {count}")
#         results = model.predict(frame, conf = 0.25)
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 # Find the center coordinates of the bouding boxes
#                 cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
#                 #Find the height and width of the bounding boxes
#                 bbox_width = abs(x1 - x2)
#                 bbox_height = abs(y1 - y2)
#                 xcycwh = [cx, cy, bbox_width, bbox_height]
#                 xywh_bboxs.append(xcycwh)
#                 conf = math.ceil(box.conf[0] * 100)/100
#                 confs.append(conf)
#                 classNameInt = int(box.cls[0])
#                 oids.append(classNameInt)
#         xywhs = torch.tensor(xywh_bboxs)
#         confidence = torch.tensor(confs)
#         outputs = deepsort.update(xywhs, confidence, oids, frame)
#         if len(outputs) > 0:
#             bbox_xyxy = outputs[:,:4]
#             identities = outputs[:,-2]
#             classID = outputs[:,-1]
#             objectTracking.draw_boxes(frame, bbox_xyxy, identities, classID)
#         ctime = time.time()
#         fps = 1 / (ctime - ptime)
#         ptime = ctime
#         cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
#         cv2.putText(frame, f"Frame Count: {str(count)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#         cv2.imshow("Frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('1'):
#             break
#     else:
#         break
