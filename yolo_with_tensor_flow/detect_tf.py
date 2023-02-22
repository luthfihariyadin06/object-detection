#https://pysource.com
import cv2
import time
#from realsense_camera import *
from mask_rcnn import *

# Load Realsense camera
#rs = RealsenseCamera()
mrcnn = MaskRCNN()

#Opening Camera
device_id = 8
winname = 'Camera Output'
capture = cv2.VideoCapture(device_id)

# FPS const
font_face = cv2.FONT_HERSHEY_SIMPLEX
previous_time = 0
new_time = 0

org = (0, 50)
font_scale = 0.5
font_color = (90, 252, 3)
font_thickness = 1
font_line_type = cv2.LINE_AA
font_bottom_left_origin = False

while True:
	# Get frame in real time from Realsense camera
	#ret, bgr_frame, depth_frame = rs.get_frame_stream()
	retval, image = capture.read()
	height, width, channels = image.shape
	    
	
	# Get object mask
	#boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)
	boxes, classes, contours, centers = mrcnn.detect_objects_mask(image)

	# Draw object mask
	#bgr_frame = mrcnn.draw_object_mask(bgr_frame)

	# Show depth info of the objects
	#mrcnn.draw_object_info(bgr_frame, depth_frame)
	mrcnn.draw_object_info_new(image)

	new_time = time.time()
	fps = round(1/(new_time-previous_time),2)
	previous_time = new_time

	fps_str = str(fps)
	#cv2.putText(bgr_frame, "FPS: " + fps_str, org, font_face, font_scale, font_color, font_thickness, font_line_type, font_bottom_left_origin)
	cv2.putText(image, "FPS: " + fps_str, org, font_face, font_scale, font_color, font_thickness, font_line_type, font_bottom_left_origin)

	#cv2.imshow("depth frame", depth_frame)
	#cv2.imshow("Bgr frame", bgr_frame)
	cv2.imshow(winname, image)

	key = cv2.waitKey(1)
	if key == 27:
		print(f"Key {key} is pressed. Proceed to exit")
		break

#rs.release()
cv2.destroyAllWindows()
