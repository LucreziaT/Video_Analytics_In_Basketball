# Video_Analytics_In_Basketball
Extraction of 3 basic analytics: 
- Run a people detector on the whole video. Write on the output video the number of people detected at each frame in the scene. Qualitatively evaluate the accuracy of the detector when reporting the results. 
- Choose one person in the scene (either player or referee) and try to track him. Plot the trajectory on the output video. Qualitatively evaluate the obtained results.
- Detect how many times the ball possession change (e.g. all 10 players go from one half of the court to the other one). Display the result on the output video.
# How does it work
The final version is the "player_detection_tracking_ball_possession.py", to make it compilate it's necessary to install: 
* numpy
* cv2
#
For the detector i used YOLO with the framework of OpenCV 3.2, this way there is no need to install anything, just download the weights at the link: https://drive.google.com/file/d/1SzCtYkHB9rtrjRiHSAfqDXPhqNITMWre/view?usp=sharing 
and follow the video on youtube if you have any doubt https://www.youtube.com/watch?v=h56M5iUVgGs .
# Work Flow
* I apply a mask to the video in order to consider only the field, cutting off spectators and cameramen.
 *I use YOLO to make the detection of players, I set a control that identifies only objects classified as people, I count them and print the number on the screen, considering that in a game there are 10 players and 3 referees the number that should be obtained is 13.
Unfortunately, one referee is too far from the camera and the other two go in and out often, so the algorithm never reaches this number. 
* Using a tracking algorithm I track the movements of a player and draw a line following the displacement of the center of the bounding box, when there start to be too many occlusions the algorithm makes a mistake and starts to identify movements of different players. 
* I calculate the number of players on either side of the field, if more than 5 players are tracked and they are all on one side of the field, a ball possession check is done, if a change is identified the ball pass counter increases. Initially the ball is of the blacks, so the number of ball changes is 3, as the algorithm calculates. 
# Other methods used 
I also tried to use  Blob detection, blob detection methods are aimed at detecting regions in a digital image that differ in properties, such as brightness or color, compared to surrounding regions. 


The detection with this method has a Success rate of 40%, using YOLO I achived 52% with an accuracy of 0.5. 
For the tracking i tried to followe the center of the blob, it works but not perfectly, it a blob is not present in a frame the position start to be randomly until the original blob returns.

I also tried to use the countours for the detection, but the Success rate was 11%.

# Background Subtraction
I tried to add to the the backgroun subtraction to all of this methods, in YOLO, it doesn't work properly and decreases the performance  of the algorithm, this bacause YOLO is trained on real images of persons, with real background, not with black background, What i should do to improve the algorithm is to train a network by my self, this require lot of time and a powerfull GPU, but performance in detection would improve massively! 

In the other two cases it helps a lot with the performance, these photos prove it:

Blob Detection without Background Subtraction 
![conbackground](https://user-images.githubusercontent.com/44268830/121672061-a1593e00-caaf-11eb-9375-76a6e713126a.png)
Blob Detection with Background Subtraction 
![provo1](https://user-images.githubusercontent.com/44268830/121672076-a74f1f00-caaf-11eb-8cd5-1b8e3cbac012.png)
What the alghoritm actually sees!
![provo2](https://user-images.githubusercontent.com/44268830/121672443-188ed200-cab0-11eb-8526-5c3189f0ebfd.png)

# Output video
With background subtraction algorithm for the tracking
https://drive.google.com/file/d/1sUuKK8e6Ejy6e9CC5hARWYOCv3GWBxxs/view?usp=sharing 
Without background subtraction algorithm for the tracking
https://drive.google.com/file/d/1xQleaPV_H3OHAtUAa7KOjbI-mI2HBs-p/view?usp=sharing 
