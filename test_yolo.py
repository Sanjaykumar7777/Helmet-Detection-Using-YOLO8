import cv2
from ultralytics import YOLO


model = YOLO("best.pt")  # Ensure "best.pt" is the saved weights from training

# Open a video file
cap = cv2.VideoCapture("input_video.mp4") 

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        bbox = result['bbox']
        class_id = result['class']
        label = result['name']
        color = (0, 255, 0) if label == 'helmet' else (255, 0, 0)
        
        # Draw bounding boxes and labels on the frame
        cv2.rectangle(frame, bbox[0:2], bbox[2:4], color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)


cap.release()
out.release()
print("Video saved with bounding boxes and labels.")
