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
* A mask is applied to the video in order to consider only the field, cutting off spectators and cameramen.
* I use YOLO to make the detection of players, I set a control that identifies only objects classified as people, I count them and print the number on the screen, considering that in a game there are 10 players and 3 referees the number that should be obtained is 13. Unfortunately, one referee is too far from the camera and the other two go in and out often, so the algorithm never reaches this number. 
* I used the MIL tracker for the tracking, it's also possible to used others like: BOOSTING, KCF, TLD, MEDIANFLOW, CSRT, MOSSE etc.. The algorithm track the movements of a player and draw a line following the displacement of the center of the bounding box, when there start to be too many occlusions the algorithm makes mistakes and starts to follow movements of different players. I applied a background subtraction algorithm to the frames to which the tracking is then applied, in this way there is no confusion with the pixels of the background, this makes it difficult for the tracking to miss the player, but often if it does not find a player with the same uniform, it tends to get lost and stop in the background. If the background subtraction algorithm is not used, this problem does not arise, but often the tracker changes players with different jerseys. 
* I calculate the number of players on either side of the field, if more than 5 players are tracked and they are all on one side of the field, a ball possession check is done, if a change is identified the ball pass counter increases. Initially the ball is of the blacks, so the number of ball changes is 3, as the algorithm calculates. 
# Other methods used 
I also tried to use  Blob detection, blob detection methods are aimed at detecting regions in a digital image that differ in properties, such as brightness or color, compared to surrounding regions. 


The detection with this method has a Success rate of 40%, using YOLO I achived 52% with an accuracy of 0.5. 
For the tracking i tried to followe the center of the blob, it works but not perfectly, it a blob is not present in a frame the position start to be randomly until the original blob returns.

I also tried to use the countours for the detection, but the Success rate was 11%.

# Background Subtraction
I have tried adding the background subtraction made by me to all these methods. First, I average all the frames to get what comes closest to a background. Obviously the billboards that keep changing will continue to be in the frames. I subtract the image from each frame, apply a binary mask and morphological transformations( erosion and dilation) to better define the figures, crop the field, and then do an end with the current frame to display the players.
With YOLO, adding this technique does not work properly and decreases the performance of the algorithm, this is because YOLO is trained on real images of people, with real background, not black background. What I should do to improve the algorithm is to train a network by myself, this takes a lot of time and a powerful GPU, but the performance in detection would improve greatly! 
On the other two methods the improvements are huge, these photos prove it:

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

# Conclusions 
The detection of players in making an average over all frames has an accuracy of 52\%, the players present are calculated based on the number of bounding boxes present in the frame, sometimes when players are very close and the bounding box exceeds a certain area, the number of players is doubled. This creates an overestimation in some frames, but in most cases the calculation is correct. 
As for the tracking, at the beginning of the video it is possible to draw the bounding box that will be considered, it seems to work relatively well at first, following the player correctly even during very fast movements. When there are too many occlusions, the bounding box gets lost and starts following different players orreferees. 
As for counting changes in ball possession, the value is correct and seems to always work correctly. To improve the detection it would be necessary to train an own neural network with the images subjected to backgroun subraction. 
For the tracking I could combine an algorithm of identification of the team of the players based on the color of the shirt, so that the bounding box at least does not start tracking the movements of referees or players of another team. 
