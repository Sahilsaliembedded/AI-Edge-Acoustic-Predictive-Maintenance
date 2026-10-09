import serial
import time

PORT = "COM8"
BAUD = 921600

ser = serial.Serial(PORT, BAUD, timeout=2)

print("Connected to", PORT)

time.sleep(3)

ser.reset_input_buffer()

print("Sending R...")
ser.write(b"R")
ser.flush()

start = time.time()
total = 0

while time.time() - start < 15:

    data = ser.read(4096)

    if data:
        total += len(data)
        print("Received:", total, "bytes")

ser.close()
