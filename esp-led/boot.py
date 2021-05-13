try:
    import usocket as socket
except:
    import socket

from machine import Pin
import network

import esp
esp.osdebug(None)

import gc
gc.collect()


def connect():
    ip        = '192.168.1.99.'
    subnet    = '255.255.255.0'
    gateway   = '192.168.1.1'
    dns       = '192.168.1.1'
    ssid      = "<AP_SSID>"
    password  =  "<AP_PASSWORD>"

    station = network.WLAN(network.STA_IF)

    if station.isconnected() == True:
        print("Already connected")
        return

    station.active(True)
    station.ifconfig((ip,subnet,gateway,dns))
    station.connect()

    while station.isconnected() == False:
        pass

    print("Connection successful")
    print(station.ifconfig())

def disconnect():
    station = network.WLAN(network.STA_IF)
    station.disconnect()
    station.active(False)

red_led = Pin(2, Pin.OUT)
green_led = Pin(3, Pin.OUT)
