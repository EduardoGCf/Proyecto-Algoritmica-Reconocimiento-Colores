import cv2
import numpy as np
import serial
import time
import keyboard

# Inicializar la comunicación serial con Arduino
ser = serial.Serial('COM8', 9600, timeout=1)  # Reemplaza 'COM8' con tu puerto de Arduino
time.sleep(2)  # Permitir tiempo para que la conexión serial se inicialice

# Funciones de preprocesamiento y modelo entrenado (del código anterior)
def scale_RGB(x):
    return x * 1.0 / 255

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def trainRGB():
    X = np.array([
        [scale_RGB(250), scale_RGB(181), scale_RGB(127)], [1, scale_RGB(165), 0], [1, scale_RGB(165), 0],  # Orange
        [scale_RGB(207), scale_RGB(75), scale_RGB(65)], [scale_RGB(173), scale_RGB(216), scale_RGB(230)], [0, 0, scale_RGB(139)],  # Blue
        [scale_RGB(144), scale_RGB(238), scale_RGB(144)], [0, scale_RGB(100), 0],  # Green
        [scale_RGB(239), scale_RGB(239), scale_RGB(74)], [scale_RGB(242), scale_RGB(242), scale_RGB(62)],  # Yellow
        [scale_RGB(211), scale_RGB(211), scale_RGB(211)], [scale_RGB(169), scale_RGB(169), scale_RGB(169)],  # Grey
        [scale_RGB(0), scale_RGB(0), scale_RGB(0)], [scale_RGB(255), scale_RGB(255), scale_RGB(255)]  # Black and White
    ])
    
    y = np.array([
        [1, 0, 0],  # Orange
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0],  # Blue
        [0, 1, 0]   # Blue
    ])
    
    np.random.seed(1)
    syn0 = 2 * np.random.random((3, 3)) - 1
    syn1 = 2 * np.random.random((3, 3)) - 1

    for j in range(60000):
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        l2_error = y - l2

        if (j % 10000) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))

        l2_delta = l2_error * sigmoid(l2, deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    return syn0, syn1

def recognize_color(r, g, b, syn0, syn1):
    X = np.array([[scale_RGB(r), scale_RGB(g), scale_RGB(b)]])
    l1 = sigmoid(np.dot(X, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2

# Entrenar el modelo
syn0, syn1 = trainRGB()

# Capturar video desde la cámara
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (frame_width // 2, frame_height // 2)
movement_threshold = 80  # Ajusta este valor para establecer la sensibilidad del movimiento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el cuadro a un tamaño más pequeño
    resized_frame = cv2.resize(frame, (100, 100))

    # Obtener el color promedio del cuadro
    avg_color_per_row = np.average(resized_frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = avg_color.astype(int)

    r, g, b = avg_color[2], avg_color[1], avg_color[0]  # OpenCV usa BGR en lugar de RGB

    # Reconocer el color
    color_probabilities = recognize_color(r, g, b, syn0, syn1)
    color_index = np.argmax(color_probabilities)
    colors = ['Orange', 'Blue', 'Green', 'Yellow', 'Grey', 'Black', 'White']
    recognized_color = colors[color_index]

    # Mostrar el color reconocido en el cuadro
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, recognized_color, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Color Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Leer y mostrar los datos seriales de Arduino
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)

cap.release()
cv2.destroyAllWindows()
ser.close()
