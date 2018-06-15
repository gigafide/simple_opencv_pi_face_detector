#import required libraries
import time
import cv2

#if using the picamera, import those libraries as well
from picamera.array import PiRGBArray
from picamera import PiCamera

#point to the haar cascade file in the directory
cascPath = "haarcascade.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#start the camera and define settings
camera = PiCamera()
camera.resolution = (320, 240) #a smaller resolution means faster processing
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

#give camera time to warm up
time.sleep(0.1)

# start video frame capture
for still in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# take the frame as an array, convert it to black and white, and look for facial features
	image = still.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		image,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize=(30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	#for each face, draw a green rectangle around it and append to the image
	for(x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)

	#display the resulting image
	cv2.imshow("Display", image)

	# clear the stream capture
	rawCapture.truncate(0)

	#set "q" as the key to exit the program when pressed
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
