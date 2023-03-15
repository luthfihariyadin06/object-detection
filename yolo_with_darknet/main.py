from device_camera import *
from darknet_object_detection import *
import time
#import serial

net = DarknetDNN()
camera = DeviceCamera()
#port = '/dev/ttyUSB0'
#ser = serial.Serial(port, 9600, timeout=1)
#ser.reset_input_buffer()
timestamp = time.time()

#print(tuple(net.get_output_layers()))

while True:
    #Get frame from camera
    frame = camera.get_frame()

    net.detect_object(frame)
    #net.detect_object_rcnn(frame)
    net.draw_object(frame)

    if time.time() - timestamp >= 0.5:
        #print(net.get_command())
        direct = net.get_command()
        
        if direct == 'Right':
            command = '1'
        elif direct == 'Left':
            command = '2'
        elif direct == 'Center':
            command = '3'
        else:
            command = '0'
        
        ser.write(command.encode("utf-8"))
        line = ser.readline().decode("utf-8").rstrip()
        print(line)
        timestamp = time.time()

    #Draw grid
    camera.create_grid()

    #display the image
    camera.show()

    #exit condition
    key = cv2.waitKey(1)
    if key == 27:
        print(f"Key {key} is pressed.")
        break

camera.release()