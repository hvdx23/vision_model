import cv2

def draw_boxes(frame, boxes, confidences, class_ids, class_names):
    for i, box in enumerate(boxes):
        (x, y, w, h) = box
        color = (0, 255, 0)  # You can customize this
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
