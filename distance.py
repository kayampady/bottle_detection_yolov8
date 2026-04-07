from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best.pt")

# Known values (adjust properly)
KNOWN_WIDTH = 7  # cm (measure your bottle)
FOCAL_LENGTH = 640  # change after calibration

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    annotated_frame = frame.copy()
    bottle_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])

        # Check class name (IMPORTANT)
        if model.names[cls] == "bottles":

            bottle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Pixel width
            pixel_width = x2 - x1

            # Avoid division by zero
            if pixel_width > 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
            else:
                distance = 0

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with distance
            cv2.putText(
                annotated_frame,
                f"{distance:.2f} cm",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # Show total count
    cv2.putText(
        annotated_frame,
        f"Bottle Count: {bottle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display frame
    cv2.imshow("Detection + Distance", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()