# Sensor Fusion Nanodegree

## Project 3 - Object Tracking in 3D Space

![][img01x]

This project implements lidar-based and camera-based time-to-collision estimation methods. Several feature detectors and feature descriptors from the OpenCV library are compared to each other in terms of estimation performance. 

|Type	|Method	|
|---	|---	|
|Detector	|'AKAZE', 'BRISK', 'HARRIS', 'ORB', 'SHITOMASI', 'SIFT', 'SURF'	|
|Descriptor	|'AKAZE', 'BRIEF', 'BRISK', 'FREAK', 'ORB', 'SIFT', 'SURF'		|

Performance parameters such as number of keypoints, processing time, number of features matches and estimated time-to-collision were written to a json file for each pairing of detectors and descriptors. Performance evaluation was done in the Jupyter Notebook _./report/performance_evaluation.ipynb_.

---

[//]: # (Image References)

[img01x]: ./img/camera_frames.gif " "
[img02x]: ./img/ttc-liadar-no-outlier-rejection.png " "
[img01]: ./img/img01.png " "
[img02]: ./img/img02.png " "
[img03]: ./img/img03.png " "
[img04]: ./img/img04.png " "
[img05]: ./img/img05.png " "
[img06]: ./img/img06.png " "
[img07]: ./img/img07.png " "
[img08]: ./img/img08.png " "
[img09]: ./img/img09.png " "
[img10]: ./img/img10.png " "
[img11]: ./img/img11.png " "
[img12]: ./img/img12.png " "
[img13]: ./img/img13.png " "
[img14]: ./img/img14.png " "
[img15]: ./img/img15.png " "
[img16]: ./img/img16.png " "
[img17]: ./img/img17.png " "
[img18]: ./img/img18.png " "
[img19]: ./img/img19.png " "
[img20]: ./img/img20.png " "

### FP.1 Match 3D Objects

_Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property)._

The matching method takes the first bounding box in the prevFrame object and finds all keypoints which lie within it. Then it iterates over all bounding boxes in the currFrame object and uses the known feature corresopondences to count the number of matches in each box. The best match between two bounding boxes is detemined by the highest number of feature matches they share.    

Source code:
_/src/canFusion_Student.cpp /  matchBoundingBoxex(...) <br/>line 148ff:

### FP.2 Comput Lidar-based TTC

_Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors._

The lidar-based method uses the distance to the closest lidar point on the preceding car in the previous laser scan and the current one. Outliers are removed by calculating the distances between neighbouring points along the x-axis, and rejecting points if the distance to their closest neigbour is larger than two times the median. This technique achieved good results with the provided laser data. 

Source code:
_/src/canFusion_Student.cpp / computeLidarTTC(...)_<br/>line 192ff:

_/src/canFusion_Student.cpp / removeLidarOutliers(...)_<br/>line 214ff:


### FP.3 Associate Keypoint Correspondences with Bounding Boxes

_Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box._

In this part of the code valid keypoint matches between the previous frame and the current frame are determined. The algorithm uses two methods to reject outliers which take place in the functions _clusterKptMatchesWithROI_ and _computeTTCCamera_. In the first one, all available keypoint matches are evaluated and the a match is only used if the keypoint from the previous frame is in the coorect bounding box and the keypoint fron the current frame is in the same bounding box. This eliminates outliers which result in distances that are much lareger than the majority.   

_/src/MidTermProject_Camera_Student.cpp_ / clusterKptMatchesWithROI(...)<br/> line 242ff


### FP.4 Compute Camera-based TTC

_Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors._

The camera-based method uses the ratio of distances between two keypoints in the previous and the current frame. In the implementation, all possible combinations between two features in the same bounding box are used to calculate the distance, but they are only used if it is higher than 100 pixels. The calculation of the TTC uses the median of all distance ratios to reduce the impact of outliers.  

_/src/MidTermProject_Camera_Student.cpp_ / computeTTCCamera(...)<br/> line 264ff


### FP.5 Performance Evaluation 1

_Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened._

![][img02x]

The image above shows an estimate of the TTC which does not use outlier rejection for the lidar points. There are three outstanding points at frame 6, 11 and 16. For the validation of the calculated values the images of the lidar point clouds were used. The following table shows the minimum distance of a lidar point as determined by the code and as it was estimated from the point cloud images. 


|Frame index|xw_min (code)|TTC (code)|xw_min (from image)|TTC (from image)| 
|---|---	|---	|---	|---		|
|5	|7.58 m	|-		|7.58 m	|-			|
|6	|7.558 m|34.35 s|7.55 m	|**25.16 s**|
|...|...	|...	|...	|...		|
|10	|7.20 m	|-		|7.30 m |-			|
|11	|7.27 m	|-11.8 s|7.27 m |**24.23 s**|
|...|...	|...	|...	|...		|
|15	|6.83 m	|-		|7.93 m |-			|
|16	|6.90 m	|-9.9 s	|6.90 m |**23.1 s**	|


- Frame 6: <br/> The estimated TTC of 34.35 s seems like an outlier, however it is consistent with the point cloud image. There is no gross outlier in the lidar data.

- Frame 11: <br/> The point cloud image of frame 10 shows an outlier and the code determins the distance to the preceding car incorrectly. An evaluation of the point cloud image gives a TTC of 24.23 s instead of -11.8 s.

- Frame 16: <br/> The point cloud image of frame 15 also shows an outlier and the code determins the distance to the preceding car incorrectly. An evaluation of the point cloud image gives a TTC of 23.1 s instead of -9.9 s.

The final version of the code, however, does use accurate lidar point outlier rejection and the TTC is estimated correctly.


### FP.6 Performance Evaluation 2

_Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs._

Performance parameters such as number of keypoints, processing time, number of features matches and estimated time-to-collision were written to a json file for each pairing of detectors and descriptors. Performance evaluation was done in the Jupyter Notebook _./report/performance_evaluation.ipynb_.

#### 6.1 Graphical Evaluation

In section the results of the TTC estimation are illustrated with graphs comparing feature detector and descriptor methods.

##### 6.1.1 Number of Keypoints

The following image shows the number of keypoints detected within the bounding box for different detecting methods. It should be noted that the Harris detector is relatively incosistent over frames and only find a very low number of keypoints.

![][img01]

##### 6.1.2 Number of Keypoint Matches

The following image shows the average number of keypoint matches  within the bounding box for different descriptor methods. It should be noted that the Harris detector is relatively incosistent over frames and only find a very low number of keypoints.

![][img02]

##### 6.1.3 TTC Estimate Lidar

The following image shows an estimate of the Lidar-based TTC with outlier rejection. 

![][img03]

##### 6.1.4 TTC Estimate Camera

The following images show the estimates of Camera-based TTC for different feature detectors and descriptors. Every image shows the results of one features detector combined with all possible features descsriptors.   

##### AKAZE
The estimate is very consistent and looks plausible.

![][img04]

##### BRISK
Relatively consitsent estimates with all available feature matchers.

![][img05]

##### HARRIS
The Harris keypoint detector lead to very incosistent estimates. This is probaly due to the low number of detected features which cause the camera-based method to fail.

![][img06]

##### ORB
In the first half of the image sequence incosistent, especially in combination with Brisk, Freak and SURF descriptor.

![][img07]

##### SHITOMASI
Very consistent except in combination with the SURF descriptor.

![][img08]

##### SIFT
Relatively consistent results with all available descriptors.

![][img09]

##### SURF
The SURF keypoint detector leads to the most consistent estimates no matter which descriptor is used.

![][img10]

#### 6.2 Statistical Evaluation

This section compares statistical indicators of the TTC estimation results.

#### 6.2.1 Min / Max TTC

- Min TTC

![][img15]

- Max TTC

![][img17]

- Min / Max TTC from lowest to highest <br/> (The rankgin with min values is shorter since lines with "0.0" entries were removed.)

| | |
|---|---|
|![][img16] | ![][img18]|


#### 6.2.2 Mean TTC

The following table shows the mean TTC calculated over the image sequence for each possible combination of feature detector and feature descriptor.

- Mean TTC

![][img13]

- Mean TTC lowest to highest

![][img14]

#### 6.2.3 Standard Deviation of TTC

The standard deviation of the TTC is probably the most meaningful statistical indicator. Since the time between images is short (100 ms) and the veolicies of the cars are continuous, we can assume that they don't change too much between two time steps. Therefore the distance to the preceding car and hence the estimated TTC should not vary overly much between two images. So we can assume that a TTC estimate with low standard deviation is more accurate.

- Standard deviation of TTC

![][img11]

- Standard deviation of TTC from lowest to highest

![][img12]

## Final Thoughts

Considering the good consistency of AKAZE and the SURF feature detector which can observed in the graphical illustration as well as the standard deviation of the results i would choose one of these two if enough computational power is available.

- AKAZE - AKAZE
- SURF - SURF
- SURF - FREAK
- SURF - ORB

![][img19]

In cases where computational power is restricted i would use one of these combinations (which are 4 to 10 times faster):

- Shitomasi + OFB
- BRISK + ORB
- ORB + BRIEF

![][img20]


