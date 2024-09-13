import cv2
from detector import load_yolo_model, detect_objects
from utils import draw_boxes
import os
from tqdm import tqdm

def main():
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"

    # Verify that the files exist
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} does not exist.")
        return
    if not os.path.exists(config_path):
        print(f"Error: {config_path} does not exist.")
        return
    if not os.path.exists(names_path):
        print(f"Error: {names_path} does not exist.")
        return

    net, output_layers = load_yolo_model(weights_path, config_path)
    if net is None or output_layers is None:
        print("Failed to load YOLO model. Exiting.")
        return
    

    class_names = open(names_path).read().strip().split("\n")

    video_path = r"C:\Users\harid\Desktop\Datacom\trials\vision_model\py_model\5.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps= {fps}")
    print(f"total frames= {total_frames}")

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output_home_new.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_skip = 5  # Process every 5th frame
    frame_count = 0

    detected_objects_set = set()

    # Initialize tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                out.write(frame)  # Write the original frame to maintain video length
                pbar.update(1)
                continue

            # Reduce resolution for faster processing
            small_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

            boxes, confidences, class_ids, detected_classes = detect_objects(small_frame, net, output_layers)
            draw_boxes(small_frame, boxes, confidences, class_ids, class_names)
            
            for class_id in detected_classes:
                detected_objects_set.add(class_names[class_id])

            # Resize back to original resolution
            frame_with_boxes = cv2.resize(small_frame, (frame_width, frame_height))

            # Write the frame with detected overlays to the video file
            out.write(frame_with_boxes)

            pbar.update(1)

    cap.release()
    out.release()

    # Print unique detected objects
    print("Detected objects:", list(detected_objects_set))

if __name__ == "__main__":
    main()