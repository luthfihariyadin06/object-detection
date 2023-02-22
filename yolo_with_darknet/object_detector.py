import cv2
import time
import numpy as np

#Opening Camera
device_id = 8
winname = 'Camera Output'
capture = cv2.VideoCapture(device_id)

#Read fps
font_face = cv2.FONT_HERSHEY_SIMPLEX
previous_time = 0
new_time = 0

#FPS text const
org = (0, 50)
font_scale = 0.5
font_color = (90, 252, 3)
font_thickness = 1
font_line_type = cv2.LINE_AA
font_bottom_left_origin = False

#DNN object detection
dnn_model = "yolov3.weights"
dnn_config = "yolov3.cfg"
classes = []

net = cv2.dnn.readNet(dnn_model, dnn_config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Load coco names
with open("coco.names", "r") as f:
    classes =[line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Blob from images parameters
blob_scalefactor = 0.00392
blob_size = (320, 320)
blob_scalar = (0, 0, 0)
blob_swapRB = True
blob_crop = False
blob_ddepth = cv2.CV_32F

while(True):
    #Reading from Camera
    retval, image = capture.read()
    height, width, channels = image.shape

    #Detect object
    blob = cv2.dnn.blobFromImage(image, blob_scalefactor, blob_size, blob_scalar, blob_swapRB, blob_crop, blob_ddepth)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Show informations on display
    class_ids = []
    confidences = []
    boxes = []

    # cycle through each object outs
    for out in outs:
        for detection in out:
            scores = detection[5:]

            # the predicted class of an object is the class with highest score
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # gives a valid pass for an object with 20% or more confidence score
            # raise 0.2 for more stringent detection, lower for the other way around
            # class_id = 0 for human detection
            if confidence > 0.2 and class_id == 0:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)
                area = w * h

                # Rectangle coordinates 
                # (1.8 correcting factor to the width and height) lower to get smaller boundary box
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # removes redundant overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # draw the registered rectangle
    for i in range(len(boxes)):
        if i in indexes:
            
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # putText(img, text, org, fontFace, fontScale, color, thickness)
            cv2.putText(image, label + " " + str(round(confidence, 2)),
                        (x, y + 30), font_face, 2, color, 2)

    #Read fps
    new_time = time.time()
    fps = round(1/(new_time-previous_time),2)
    previous_time = new_time

    fps_str = str(fps)
    cv2.putText(image, "FPS: " + fps_str, org, font_face, font_scale, font_color, font_thickness, font_line_type, font_bottom_left_origin)

    #display the image
    cv2.imshow(winname, image)

    #exit condition
    key = cv2.waitKey(1)
    if key == 27:
        print(f"Key {key} is pressed.")
        break

capture.release()
cv2.destroyAllWindows()
