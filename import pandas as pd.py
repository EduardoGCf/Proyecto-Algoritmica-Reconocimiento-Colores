import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import serial
import time

# Cargar el conjunto de datos
data = pd.read_csv('ruta_al_archivo/color-classification.csv')

# Dividir las características y las etiquetas
X = data[['R', 'G', 'B']].values
y = data['Label'].values

# Escalar las características RGB
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Mapear las etiquetas a números
label_mapping = {label: index for index, label in enumerate(np.unique(y))}
y_train_mapped = np.array([label_mapping[label] for label in y_train])
y_test_mapped = np.array([label_mapping[label] for label in y_test])

# Convertir las etiquetas a one-hot encoding
y_train_one_hot = np.zeros((y_train_mapped.size, y_train_mapped.max() + 1))
y_train_one_hot[np.arange(y_train_mapped.size), y_train_mapped] = 1

# Inicializar pesos aleatorios para la red neuronal
np.random.seed(1)
syn0 = 2 * np.random.random((3, 3)) - 1
syn1 = 2 * np.random.random((3, len(label_mapping))) - 1

# Función de activación sigmoide
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Entrenamiento de la red neuronal
for j in range(60000):
    l0 = X_train
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    l2_error = y_train_one_hot - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# Función para reconocer el color
def recognize_color(r, g, b, syn0, syn1):
    rgb_scaled = scaler.transform([[r, g, b]])
    l1 = sigmoid(np.dot(rgb_scaled, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2

# Inicializar la comunicación serial con Arduino
ser = serial.Serial('COM8', 9600, timeout=1)  # Reemplaza 'COM8' con tu puerto de Arduino
time.sleep(2)  # Permitir tiempo para que la conexión serial se inicialice

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
    recognized_color = [label for label, index in label_mapping.items() if index == color_index][0]

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