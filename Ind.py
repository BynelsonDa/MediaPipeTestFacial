import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Captura de cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir BGR a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar imagen con MediaPipe
    results = hands.process(rgb)

    estado_mano = "SIN MANO"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            dedos_extendidos = 0

            # Índices de dedos (punta y articulación media)
            dedos = [
                (8, 6),   # Índice
                (12, 10), # Medio
                (16, 14), # Anular
                (20, 18)  # Meñique
            ]

            for punta, medio in dedos:
                if hand_landmarks.landmark[punta].y < hand_landmarks.landmark[medio].y:
                    dedos_extendidos += 1

            if dedos_extendidos >= 3:
                estado_mano = "MANO ABIERTA"
            else:
                estado_mano = "MANO CERRADA"

            # Dibujar landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Mostrar estado
    cv2.putText(
        frame,
        estado_mano,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("MediaPipe - Mano Abierta o Cerrada", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()