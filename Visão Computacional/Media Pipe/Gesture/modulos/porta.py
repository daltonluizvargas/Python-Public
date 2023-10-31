# pip install pyserial --user
import serial
import time

porta = 'COM3'
baud_rate = 9600

def abrir_conexao_porta():
    ser = serial.Serial(port=porta, baudrate=baud_rate, timeout=1)
    return ser