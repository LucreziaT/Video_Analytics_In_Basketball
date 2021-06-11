import numpy as np
import argparse
import cv2 as cv2

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", help="path to input video")
	ap.add_argument("-o", "--output", help="path to output image")
	ap.add_argument("-n", "--number_of_frames", type = int, help="number of frame to elaborate", default=0)
	args = vars(ap.parse_args())

	cap = cv2.VideoCapture(args["input"])
	#cap = cv2.VideoCapture( )

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("Width " + str(width))
	print("Height " + str(height))
	print("FPS " + str(cap.get(cv2.CAP_PROP_FPS)))
	video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	print("N frame video " + str(video_frames))
	video_format = cap.get(cv2.CAP_PROP_FOURCC)
	print("Video format ", end = '')
	print(video_format)

	num_frame = args["number_of_frames"]
	imgAv = np.zeros(shape=([int(height), int(width), 3]), dtype=np.uint64)

	i = 0
	if num_frame != 0:
		video_frames = num_frame
	for i in range(int(video_frames)-1):
		
		ret, frame = cap.read() # read one frame

		if not ret:
			break

		imgAv = imgAv + np.uint64(frame)

		if i % 10 == 0:
			print(str(round(i/int(video_frames)*100, 3)) + " %", end = '\r')

	imgAv = np.uint64(imgAv / i)
	cv2.imwrite(args["output"],imgAv)
	cv2.imshow('Video average', np.uint8(imgAv))
	cv2.waitKey(10000)
	cap.release()
	cv2.destroyAllWindows()