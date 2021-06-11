
import cv2
import numpy as np;

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

def Transformation(frame, background,kernel1,kernel2,pts): #mog2  #putText(fram,f'testo{}{}'(x,y).font,1,(0,0,0),2,cv2Line.4) font=cv2.FONT_HERSHEY_SIMPLEX our=cv2.VideoWriteout.write(frame)
    diff = cv2.absdiff(frame,background)
    finale = cv2.bitwise_and(diff,frame)
    img2gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    ret2, mask = cv2.threshold(img2gray,25,255,cv2.THRESH_BINARY)
    mask=cv2.erode(mask,kernel1,iterations = 1)
    mask = cv2.dilate(mask,kernel2,iterations = 1)
    finale = cv2.bitwise_and(frame,frame, mask=mask)
    solocampo = createMask(frame, pts)
    finalevero = cv2.bitwise_and(finale, solocampo)
    return finalevero

#open the video path

nframe= 2038
frame1=[]
cap = cv2.VideoCapture("/home/lucrezia/Desktop/Video_Analytics_In_Basketball/CV_basket.mp4")
background = cv2.imread("/home/lucrezia/Desktop/Video_Analytics_In_Basketball/im_out.png")
frame_w=int(cap.get(3))
frame_h=int(cap.get(4))
kernel1=np.ones((2,2),np.uint8)
kernel2=np.ones((6,6),np.uint8)
countertotale=0
#pts = np.array([[62,662],[707,761],[1367,638],[1031,545],[692,569],[341,551]], np.int32) # points to cut out the important part of image
pts = np.array([[14,654],[416,747],[671,740],[718,739],[801,752],[1376,633],[1064,521],[608,534],[333,518]], np.int32)
pts = pts.reshape((-1,1,2)) # change array formatq
ret,frame = cap.read()
finalevero=Transformation(frame,background,kernel1,kernel2,pts)
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 25, (frame_w,frame_h))
for i in range(nframe):
	counter=0
	boundingbox=[]
	print('frame: ',i)
    #capture frame by fram
	ret,frame = cap.read()
    #difference between background and actual fram
	finalevero=Transformation(frame,background,kernel1,kernel2,pts)

	edged = cv2.Canny(finalevero, 30, 200)
	edged2=edged.copy()
    #cv2.imshow('Canny Edges After Contouring', edged)
    #cv2.waitKey(1)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the imag
	contours, hierarchy = cv2.findContours(edged2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
	for s,c in enumerate(contours):
		x,y,w,h  = cv2.boundingRect(c)
		centrox=(x+x+w)/2
		centroy=(y+y+h)/2
        #print('area: ',(w*h)/2)
		if cv2.contourArea(c) >=800 and cv2.contourArea(c)<=3500 and h>=w:
			boundingbox.append(c)
			len(boundingbox)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
			counter+=1
	print('counter: ',counter)
	countertotale=countertotale+counter
	successdetect=((countertotale)/(13*(i+1)))*100
	print(successdetect)
    #print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
	#cv2.putText(frame, f'CounterBallPossession: {counterballchanges}', (50,180), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(frame, f'PlayerDetected: {counter}', (50,210), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(frame, f'SuccDetect: {int(successdetect)}', (50,240), font,1, (255,255,255), 2, cv2.LINE_4)
	cv2.putText(frame, f'frame: {i}', (50,270), font,1, (255,255,255), 2, cv2.LINE_4)
	out.write(frame)
	cv2.imshow('Contours', frame)
	cv2.waitKey(1)


    #cv2.imshow("Keypoints", im_with_keypoints)
    #cv2.waitKey(1)
	if cv2.waitKey(1) & 0xff == ord('q'):
        #print('frame1: ',frame1)
		break
print('FINAL SUCCDECT= ',int(successdetect))
print('AVPLAYER: ',int(countertotale/nframe))
cap.release()
out.release()
cv2.destroyAllWindows()
