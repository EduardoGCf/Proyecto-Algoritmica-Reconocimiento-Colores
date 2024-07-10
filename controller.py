import serial
import time
import keyboard

# Replace 'COM3' with the port your Arduino is connected to
ser = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # Allow some time for the serial connection to initialize

print("Press arrow qkeys to control the servo. Press 'q' to quit.")

while True:
    if keyboard.is_pressed('right'):
        ser.write(b'f')  # Forward (clockwise)
        print("Moving forward")
        time.sleep(0.1)
        ser.write(b's')  # Stop
        
    elif keyboard.is_pressed('left'):
        ser.write(b'b')  # Backward (counterclockwise)
        print("Moving backward")
        time.sleep(0.1)
        ser.write(b's')  # Stop
        
    elif keyboard.is_pressed('s'):
        ser.write(b's')  # Stop
        print("Stopping")
        time.sleep(0.1)
  
    elif keyboard.is_pressed('up'):
        ser.write(b'u')  # Increase angle for regular servo
        print("Regular servo moving up")
        time.sleep(0.1)
        
    elif keyboard.is_pressed('down'):
        ser.write(b'd')  # Decrease angle for regular servo
        print("Regular servo moving down")
        time.sleep(0.1)
        
    elif keyboard.is_pressed('q'):
        print("Exiting")
        break

    
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        
ser.close()
