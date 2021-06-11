# Standard imports
import cv2
import numpy as np;
global blob_selected_check
from collections import deque
#Mask for cutting the field
def createMask(image, points):
	"""
	@Description: create a mask from given points
	@Parameters:
		- image -> source image, it only serves for dimensions.
		- points -> points of the polygon, the mask is created to eliminate points outside the polygon
	@Return: mask image, it's white inside the polygon and black outside. To apply the mask do an AND operation to image.
	"""
	maskImg = np.zeros(image.shape, dtype=np.uint8)
	cv2.fillConvexPoly( maskImg, points, (255,255,255) )
	return maskImg
#BackGround Subtraction
def Transformation(frame, background,kernel1,kernel2,pts1):
    diff = cv2.absdiff(frame,background)
    finale = cv2.bitwise_and(diff,frame)
    img2gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    ret2, mask = cv2.threshold(img2gray,25,255,cv2.THRESH_BINARY)
    mask=cv2.erode(mask,kernel1,iterations = 1)
    mask = cv2.dilate(mask,kernel2,iterations = 1)
    finale = cv2.bitwise_and(frame,frame, mask=mask)
    solocampo = createMask(frame, pts1)
    finalevero = cv2.bitwise_and(finale, solocampo)
    return finalevero
#Blob Selector For Tracking
def BlobSelector(blob_x, blob_y, key,frame, bool_draw_line = 1):
	print('Sono dentro')
	global blob_selected_check
	if bool_draw_line == 1:
		cv2.line(frame, (int(x_blob),int(y_blob)), (int(key.pt[0]),int(key.pt[1])), (252,15,192), 20)
	blob_x = key.pt[0]                  #New blob's X coordinate assignment
	blob_y = key.pt[1]                  #New blob's X coordinate assignment
	print('blobx: ',blob_x)
	print('bloby: ',blob_y)
	blob_selected_check = True
	return(blob_x,blob_y,blob_selected_check)

#variable to be set
nframe= 2038 #number of frame
pos=0 #0 ball to black, 1 ball to white
counterballchanges=0
countertotale=0
font = cv2.FONT_HERSHEY_SIMPLEX
frame1=[]
cap = cv2.VideoCapture("/home/lucrezia/Desktop/Video_Analytics_In_Basketball/CV_basket.mp4")
background = cv2.imread("/home/lucrezia/Desktop/Video_Analytics_In_Basketball/im_out.png")
frame_w=int(cap.get(3))
frame_h=int(cap.get(4))
track = cv2.TrackerMIL_create()
pts = deque(maxlen=124)
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 25, (frame_w,frame_h))

kernel1=np.ones((2,2),np.uint8)
kernel2=np.ones((6,6),np.uint8)
pts1 = np.array([[62,662],[696,731],[1367,638],[1031,545],[692,569],[341,551]], np.int32) # points to cut out the important part of image
pts1 = pts1.reshape((-1,1,2)) # change array format
ret,frame = cap.read()
finalevero=Transformation(frame,background,kernel1,kernel2,pts1)

blob_selected_check = False         #Flag to check if a blob was tracked in the last frame
x_blob = 0                          #Last selected blob X coordinate
y_blob = 0                          #Last selected blob Y coordinate

b_box = cv2.selectROI('Frame',frame)
track.init(frame,b_box)
for i in range(nframe):
    # capture frame by frame
	ret,frame = cap.read()
	if not ret:
		break
	succ, box = track.update(frame)

    #difference between background and actual frame
	finalevero=Transformation(frame,background,kernel1,kernel2,pts1)
	params = cv2.SimpleBlobDetector_Params()

	# Setting Parameters for Blob

    # Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;

    # Filter by Area.
	params.minDistBetweenBlobs=85
	params.filterByArea = True
	params.minArea = 120
	params.maxArea=3000

    #Filter by Circularity
	params.filterByCircularity = False
    #params.minCircularity = 0.1

    # Filter by Convexity
	params.filterByConvexity = False
    #params.minConvexity = 0.02

    # Filter by Inertia
	params.filterByInertia = False
    #params.minInertiaRatio = 0.01
	detector = cv2.SimpleBlobDetector_create(params)
	detector.empty()
	keypoints = detector.detect(finalevero)
	counter=len(keypoints)
	countertotale=countertotale+counter

	print('frame: ',i)
	print('counter: ',counter)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #im_with_keypoints = cv2.drawKeypoints(finale, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
	countdx=0
	countsx=0
	# Identifing Position of the Players for the Ball Possession
	for keypoint in keypoints:
		if keypoint.pt [0]>703:
			countdx+=1
		else:
			countsx+=1
	print('countdx: ',countdx)
	print('countsx: ',countsx)
	if countdx==counter:
		if pos==1:
			print(pos)
			counterballchanges+=1
			print('cambio palla bianchi->neri')
			pos=0
		else:
			print(pos)
			pos=0

	elif countsx==counter:
		if pos==0:
			print(pos)
			pos=1
			counterballchanges+=1
			print('cambio palla neri->bianchi')
		else:
			print(pos)
			pos=1

	if i==199:
		cv2.imwrite("lol.png", im_with_keypoints)

	successdetect=((countertotale)/(13*(i+1)))*100

	#BLOB SELECTION

    #ROI Selector
	if i == 0:
		c,r,m,n = cv2.selectROI(im_with_keypoints)
	for keypoint in keypoints:
		print('x1: ',keypoint.pt[0])
		print('y1: ',keypoint.pt[1])
		if i < 310 and blob_selected_check == False and keypoint.pt[0] > c and keypoint.pt[0] < c+m and keypoint.pt[1] > r and keypoint.pt[1] < r+n:
			x_blob,y_blob,blob_selected_check=BlobSelector(x_blob, y_blob, keypoint,im_with_keypoints, 0)
			print('X_BLOB 1:', x_blob)
			break
		if i < 310 and blob_selected_check == True and (keypoint.pt[0] - x_blob) < 20 and (keypoint.pt[1] - y_blob) < 20:
			x_blob,y_blob,blob_selected_check=BlobSelector(x_blob, y_blob, keypoint,im_with_keypoints)
			print('X_BLOB 2: ',x_blob)
			break
		else:
			if blob_selected_check == True and (keypoint.pt[0] - x_blob) < 20 and (keypoint.pt[1] - y_blob) < 20:
				x_blob,y_blob,blob_selected_check=BlobSelector(x_blob, y_blob, keypoint,im_with_keypoints)
				print('X_BLOB 2:', x_blob)
				break
			if blob_selected_check == False and (keypoint.pt[0] - x_blob) < 30 and (keypoint.pt[1] - y_blob) < 30:
				x_blob,y_blob,blob_selected_check=BlobSelector(x_blob, y_blob, keypoint,im_with_keypoints)
				print('X_BLOB 3: ',x_blob)
				break
			else:
				print('INVALID BLOB')
				blob_selected_check = False
				break
	if succ:
		(x,y,w,h) = [int(i) for i in box]
		cv2.rectangle(im_with_keypoints,(x,y), (x+w,y+h),(0,255,0),2)

		xc = int(x + w / 2)
		yc = int(y + h / 2)
		cntr = (xc, yc)

		pts.appendleft(cntr)
		for l in range(1, len(pts)):
			if pts[l - 1] is None or pts[l] is None:
				continue
			cv2.line(im_with_keypoints,pts[l - 1],pts[l],(0, 255, 0),2)




	# Creation of the ourput video
	cv2.putText(im_with_keypoints, f'CounterBallPossession: {counterballchanges}', (50,180), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(im_with_keypoints, f'PlayerDetected: {counter}', (50,210), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(im_with_keypoints, f'SuccDetect: {int(successdetect)}', (50,240), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(im_with_keypoints, f'frame: {i}', (50,270), font,1, (255,255,255), 2, cv2.LINE_4)
	out.write(im_with_keypoints)
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(1)
	if cv2.waitKey(1) & 0xff == ord('q'):
		print('frame1: ',frame1)
		break

#print(counterballchanges)
print('FINAL SUCCDECT= ',int(successdetect))
print('AVPLAYER: ',int(countertotale/nframe))
cap.release()
out.release()
cv2.destroyAllWindows()
