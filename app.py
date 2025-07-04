import cv2
from ultralytics import YOLO

# Load the model (replace with your full path if needed)
model = YOLO("weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Webcam could not be opened.")
    exit()

print("✅ Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv11 Webcam Inference", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
