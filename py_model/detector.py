# import cv2
# import numpy as np

# import cv2

# def load_yolo_model(weights_path, config_path):
#     try:
#         net = cv2.dnn.readNet(weights_path, config_path)
#         layer_names = net.getLayerNames()
#         try:
#             output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#         except TypeError:
#             output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#         return net, output_layers
#     except Exception as e:
#         print(f"Error loading YOLO model: {e}")
#         return None, None

# def detect_objects(frame, net, output_layers, conf_threshold=0.5):
#     """
#     Detects objects in a given frame using the YOLO model.
#     """
#     if frame is None:
#         print("No frame available for detection")
#         return [], [], []
    
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(output_layers)
    
#     height, width = frame.shape[:2]
#     boxes = []
#     confidences = []
#     class_ids = []

#     for detection in detections:
#         for obj in detection:
#             scores = obj[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > conf_threshold:
#                 center_x = int(obj[0] * width)
#                 center_y = int(obj[1] * height)
#                 w = int(obj[2] * width)
#                 h = int(obj[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     return boxes, confidences, class_ids

import cv2
import numpy as np

def load_yolo_model(weights_path, config_path):
    try:
        print(f"Loading YOLO model with weights: {weights_path} and config: {config_path}")
        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        except TypeError:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except IndexError:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        print("YOLO model loaded successfully.")
        return net, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None


def detect_objects(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_classes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_classes.append(class_id)

    return boxes, confidences, class_ids, detected_classes