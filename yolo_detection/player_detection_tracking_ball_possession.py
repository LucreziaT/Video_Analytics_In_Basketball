import cv2
import numpy as np
import time
from collections import deque
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

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video & background
background = cv2.imread("im_out.png")
cap = cv2.VideoCapture("CV_basket.mp4")

# Create variables
track = cv2.TrackerMIL_create()
pts = deque(maxlen=124)
frame_w=int(cap.get(3))
frame_h=int(cap.get(4))
outvideo = cv2.VideoWriter('output7.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 25, (frame_w,frame_h))
ret, frame = cap.read()
kernel1=np.ones((2,2),np.uint8)
kernel2=np.ones((7,7),np.uint8)
pts1 = np.array([[14,654],[416,747],[671,740],[718,739],[801,752],[1376,633],[1064,521],[608,534],[333,518]], np.int32) # points to cut out the important part of image
pts1 = pts1.reshape((-1,1,2))
frame2=Transformation(frame,background,kernel1,kernel2,pts1) # i don't know why it works so much worse with the background subtraction!
cv2.imshow('Frame', frame)
b_box = cv2.selectROI('Frame',frame2)
track.init(frame2,b_box)
countertotale=0
font = cv2.FONT_HERSHEY_SIMPLEX
numero=0
pos=0 #0 ball to black, 1 ball to white
counterballchanges=0
#Let's start
while True and numero!= 2241:
    #print('FRAME: ',numero)
    counter=0
    countdx=0
    countsx=0
    numero=numero+1
    #print('FRAME: ',numero)
    ret, frame = cap.read()
    if not ret:
        break
    succ, box = track.update(frame)

    frame2=Transformation(frame,background,kernel1,kernel2,pts1) # i don't know why it works so much worse with the background subtraction!
    solocampo = createMask(frame, pts1)
    frame1 = cv2.bitwise_and(frame, solocampo)
    #img = cv2.resize(frame, None, fx=0.4, fy=0.4) #if the video is too big
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #detection with yolo
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                #print('label: ',class_id)
                if class_id==0: #only class "person" is condsidered and showed in the output video
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    #print('centerx: ',center_x)
                    #print('area: ',w*h/2)
                    if center_x>703:
                        countdx+=1
                    else:
                        countsx+=1

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    counter=len(boxes)
    #print('NUmero: ',len(boxes))
    countertotale=countertotale+counter
    if counter>5:
        if countdx==counter:
            if pos==1:
                counterballchanges+=1
                cv2.putText(frame, 'BALL POSSESSION CHANGE: White->Black', (688,370), font,10, (170, 74, 68), 2, cv2.LINE_4)
                pos=0
                time.sleep(3)
            else:
                pos=0
        elif countsx==counter:
            if pos==0:
                counterballchanges+=1
                cv2.putText(frame, 'BALL POSSESSION CHANGE: Black->White ', (688,370), font, 10, (170, 74, 68), 2, cv2.LINE_4)
                pos=1
                time.sleep(3)
            else:
                pos=1

    successdetect=(countertotale/(13*numero))*100
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            #cv2.putText(frame, label, (x, y + 30), font, 3, color, 3) # no need to show the label
    if succ:
        (x1,y1,w1,h1) = [int(numero) for numero in box]
        cv2.rectangle(frame,(x1,y1), (x1+w1,y1+h1),(0,255,0),2)

        xc = int(x1 + w1 / 2)
        yc = int(y1 + h1 / 2)
        cntr = (xc, yc)

        pts.appendleft(cntr)
        for l in range(1, len(pts)):
            if pts[l - 1] is None or pts[l] is None:
                continue
            cv2.line(frame,pts[l - 1],pts[l],(0, 255, 0),2)

    cv2.putText(frame, f'BallPossessionCounter: {counterballchanges}', (50,180), font,1, (255,255,255), 2, cv2.LINE_4)
    cv2.putText(frame, f'PlayersDetected: {len(boxes)} / 13', (50,210), font,1, (255,255,255), 2, cv2.LINE_4)
    cv2.putText(frame, f'CumulativeDetectionSucc: {int(successdetect)} %', (50,240), font,1, (255,255,255), 2, cv2.LINE_4)
    #cv2.putText(frame, f'frame: {numero}', (50,270), font,1, (255,255,255), 2, cv2.LINE_4)
    cv2.imshow("Image", frame)
    outvideo.write(frame)
    cv2.waitKey(1000)
    if cv2.waitKey(1000) & 0xff == ord('q'):
        break
print('FINAL SUCCDECT= ',int(successdetect))
#print('AVPLAYER: ',int(countertotale/numero))
cap.release()
outvideo.release()
cv2.destroyAllWindows()
