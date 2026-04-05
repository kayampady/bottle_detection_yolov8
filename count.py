from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    bottle_count = 0

    annotated_frame = results[0].plot()  # YOLO draws boxes

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "bottles":
            bottle_count += 1

    # Add count text
    cv2.putText(annotated_frame, f"Count: {bottle_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)

    cv2.imshow("Bottle Detection + Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()