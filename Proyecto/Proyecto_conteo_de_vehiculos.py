import cv2
import dlib
import imutils
from imutils.video import VideoStream
import time

# Inicializa video
cap = cv2.VideoCapture("highway.mp4")  # Cambia esto por 0 si usas cámara web

# Línea de conteo (y-coordinate)
count_line_position = 250
vehicle_count = 0

# Inicializa detector de movimiento y rastreador
detector = cv2.createBackgroundSubtractorMOG2()
tracker_list = []
trackers = dlib.correlation_tracker
track_id = 0

# Umbral de detección
min_width = 80
min_height = 80
offset = 6  # margen para contar solo una vez

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = detector.apply(frame)
    dilated = cv2.dilate(fgmask, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, count_line_position), (750, count_line_position), (255, 0, 255), 2)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        validate_counter = (w >= min_width) and (h >= min_height)
        if not validate_counter:
            continue

        # Inicializa rastreador
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(x, y, x + w, y + h)
        tracker.start_track(frame, rect)
        tracker_list.append((tracker, x, y, w, h))

    # Actualiza rastreadores
    new_detects = []
    for tracker_data in tracker_list:
        tracker, x, y, w, h = tracker_data
        tracker.update(frame)
        pos = tracker.get_position()
        cx, cy = center_handle(int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height()))

        if count_line_position - offset < cy < count_line_position + offset:
            vehicle_count += 1
            tracker_list.remove(tracker_data)

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Muestra resultados
    cv2.putText(frame, "Vehiculos: " + str(vehicle_count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Contador de Vehículos", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
