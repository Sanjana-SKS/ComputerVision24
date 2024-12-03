#Version: 1.0
#Python 3.11.0
#Authors: Matthew, Sanjana, Judy
#Date: 12/2/2024
#Description: This file contains the VolumeReader class which takes serial data from an arduino
import serial
import threading
import re
#Might need to change port and baud rate to match the arduino
class VolumeReader:
    def __init__(self, port='COM6', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        try:
            self.serial_connection = serial.Serial(port, baud_rate, timeout=1)
            print(f"Connected to {port} at {baud_rate} baud.")
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            self.serial_connection = None
        self.volume = 0
        self.running = True
        self.thread = threading.Thread(target=self.read_volume)
        self.thread.start()
        
    def read_volume(self):
        volume_pattern = re.compile(r"(\d+\.?\d*)")  
        while self.running and self.serial_connection:
            if self.serial_connection.in_waiting > 0:
                try:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    match = volume_pattern.search(line)
                    if match:
                        self.volume = float(match.group(1))
                except Exception as e:
                    print(f"Error reading volume: {e}")

    def get_volume(self):
        return self.volume

    def stop(self):
        self.running = False
        if self.serial_connection:
            self.serial_connection.close()
        self.thread.join()