import numpy as np
import cv2
import serial
import time
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras


from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

import cv2


# Parameters
img_width, img_height = 100, 100
batch_size = 32

# Initialize ImageDataGenerator for loading images from directories
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    'path_to_your_dataset',  # Replace with the path to your dataset
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path_to_your_dataset',  # Replace with the path to your dataset
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Extract class labels
class_labels = list(train_generator.class_indices.keys())

# Load data
X_train, y_train = [], []
for _ in range(len(train_generator)):
    x, y = train_generator.next()
    X_train.extend(x)
    y_train.extend(y)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test, y_test = [], []
for _ in range(len(validation_generator)):
    x, y = validation_generator.next()
    X_test.extend(x)
    y_test.extend(y)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Scale features
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Map labels to numbers
y_train_mapped = np.argmax(y_train, axis=1)
y_test_mapped = np.argmax(y_test, axis=1)

# Initialize random weights for the neural network
np.random.seed(1)
syn0 = 2 * np.random.random((X_train_scaled.shape[1], 3)) - 1
syn1 = 2 * np.random.random((3, len(class_labels))) - 1

# Sigmoid activation function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Train the neural network
for j in range(60000):
    l0 = X_train_scaled
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    l2_error = y_train - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# Function to recognize color
def recognize_color(image, syn0, syn1):
    image_resized = cv2.resize(image, (img_width, img_height))
    image_scaled = image_resized.reshape((1, -1))
    image_scaled = scaler.transform(image_scaled)
    l1 = sigmoid(np.dot(image_scaled, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2

# Initialize serial communication with Arduino
ser = serial.Serial('COM8', 9600, timeout=1)  # Replace 'COM8' with your Arduino port
time.sleep(2)  # Allow time for the serial connection to initialize

# Capture video from the camera
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (frame_width // 2, frame_height // 2)
movement_threshold = 80  # Adjust this value to set the sensitivity of the movement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Recognize the color
    color_probabilities = recognize_color(frame, syn0, syn1)
    color_index = np.argmax(color_probabilities)
    recognized_color = class_labels[color_index]

    # Display the recognized color in the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, recognized_color, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Color Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read and display serial data from Arduino
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)

cap.release()
cv2.destroyAllWindows()
ser.close()
