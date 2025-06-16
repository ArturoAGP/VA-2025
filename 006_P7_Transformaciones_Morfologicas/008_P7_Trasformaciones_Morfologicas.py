import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

# Nuevos rangos para rojo en HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear dos máscaras para cubrir todo el rango de rojo
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combinar máscaras
    mask = cv2.bitwise_or(mask1, mask2)

    # Aplicar la máscara sobre la imagen original
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Kernel para operaciones morfológicas
    kernel = np.ones((15,15), np.uint8)

    # Filtros
    smoothed = cv2.filter2D(res, -1, kernel)
    blur = cv2.GaussianBlur(res, (5,5), 0)
    median = cv2.medianBlur(res, 5)
    bilateral = cv2.bilateralFilter(res, 5, 10, 10)

    # OPERACIONES MORFOLÓGICAS: TOPHAT y BLACKHAT
    tophat = cv2.morphologyEx(res, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(res, cv2.MORPH_BLACKHAT, kernel)

    # Mostrar ventanas
    cv2.imshow('Original', frame)
    #cv2.imshow('Blur', blur)
    #cv2.imshow('Median', median)
    #cv2.imshow('Bilateral', bilateral)
    cv2.imshow('Top Hat', tophat)
    cv2.imshow('Black Hat', blackhat)

    # Salida con ESC
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
