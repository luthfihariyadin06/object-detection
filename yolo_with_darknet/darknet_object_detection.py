import cv2
import numpy as np

class DarknetDNN:
    def __init__(self, dnn_model = "yolo_with_darknet/yolov3-tiny.weights", dnn_config = "yolo_with_darknet/yolov3-tiny.cfg"):
        #Initiate DNN model using Darknet framework
        print("Initiating Darknet ...")
        self.dnn_model = dnn_model
        self.dnn_config = dnn_config
        self.net = cv2.dnn.readNet(self.dnn_model, self.dnn_config)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.classes = []

        with open("yolo_with_darknet/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        #Blob parameter
        self.blob_scalefactor = 0.00392
        self.blob_size = (320, 320)
        self.blob_scalar = (0, 0, 0)
        self.blob_swapRB = True
        self.blob_crop = False
        self.blob_ddepth = cv2.CV_32F

        #Threshold for detecting object
        self.detection_threshold = 0.2

    def detect_object(self, image):
        #Set image into blob
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, self.blob_scalefactor, self.blob_size, self.blob_scalar, self.blob_swapRB, self.blob_crop, self.blob_ddepth)

        #Add blob as input and wait for output
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        #Object informations
        self.object_boxes = []
        self.object_classes = []
        self.object_centers = []
        self.object_contours = []
        self.object_positions = []
        self.object_confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]

                class_id = np.argmax(scores)
                confidance = scores[class_id]
                
                if confidance < self.detection_threshold:
                    continue
                
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x1 = int(cx - w/1.8)
                y1 = int(cy - h/1.8)
                x2 = int(cx + w/1.8)
                y2 = int(cy + h/1.8)

                self.object_boxes.append([x1, y1, x2, y2])
                self.object_classes.append(class_id)
                self.object_centers.append([cx, cy])
                self.object_confidences.append(confidance)
                #print(f"Conf = {confidance} and object is : {self.classes[class_id]}")

        #return self.object_boxes, self.object_classes

    def detect_object_rcnn(self, image):
        #Set image into blob
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, self.blob_scalefactor, self.blob_size, self.blob_scalar, self.blob_swapRB, self.blob_crop, self.blob_ddepth)

        #Add blob as input and wait for output
        self.net.setInput(blob)
        boxes, masks = self.net.forward(self.output_layers)
        #boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        print(masks.shape)
        print(boxes.shape)
        detection_count = boxes.shape[1]
        

        #Object informations
        self.object_boxes = []
        self.object_classes = []
        self.object_centers = []
        self.object_contours = []
        self.object_positions = []
        self.object_confidences = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            
            if score < self.detection_threshold:
                continue

            x1 = int(box[3] * width)
            y1 = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.object_boxes.append([x1, y1, x2, y2])
            self.object_confidences.append(score)
            self.object_centers.append([cx, cy])
            self.object_classes.append(class_id)


        
    
    def draw_object(self, frame):
        indexes = cv2.dnn.NMSBoxes(self.object_boxes, self.object_confidences, 0.4, 0.3)
        for i, box, class_id, center in zip(range(len(self.object_boxes)), self.object_boxes, self.object_classes, self.object_centers):
            if i not in indexes:
                continue

            x1, y1, x2, y2 = box
            cx, cy = center
            name = self.classes[class_id]
            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.circle(frame, (cx, cy), 10, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, name.capitalize(), (x1 + 5, y1 + 25), 0, 0.8, (255, 255, 255), 2)


    def get_classes(self):
        return self.classes
    
    def get_layer_names(self):
        return self.layer_names
    
    def get_output_layers(self):
        return self.output_layers
    
    def get_colors(self):
        return self.colors