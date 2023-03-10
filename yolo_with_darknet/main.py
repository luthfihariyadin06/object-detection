from device_camera import *
from darknet_object_detection import *

net = DarknetDNN()
camera = DeviceCamera()

#print(tuple(net.get_output_layers()))

while True:
    #Get frame from camera
    frame = camera.get_frame()

    net.detect_object(frame)
    #net.detect_object_rcnn(frame)
    net.draw_object(frame)

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