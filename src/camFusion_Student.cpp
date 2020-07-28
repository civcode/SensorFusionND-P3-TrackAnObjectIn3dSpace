
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include "properties.h"
#include "mycvtools.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 0);
    cv::resizeWindow(windowName, 400, 400);
    cv::moveWindow(windowName, 50, 50);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(10); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    bool use_graphical_output = false;
    // for (auto match : matches) {
    //     cout << "[query, train]: [" << match.queryIdx << ", " << match.trainIdx << "]" << endl; 
    // }

    // algorithm
    // over all roi in prevFrame
    //      find all matched keypoints in the roi
    //      for every kp
    //          for every roi in currFrame
    //              if roi contains corresponding kp
    //                  increase count
    //          end
    //      end
    //      find roi with highest count

    vector<cv::Scalar> colors { {255,0,0,1}, {0,255,0,1}, {0,0,255,1}, 
                                {255,255,0,1}, {0,255,255,1}, {255,0,255,1}};

    

    cv::Mat matchImg;// = currFrame.cameraImg.clone();

    if (use_graphical_output)
        mycv::drawMatches(prevFrame.cameraImg, prevFrame.keypoints,
                        currFrame.cameraImg, currFrame.keypoints,
                        matches, matchImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


    // for (auto box : prevFrame.boundingBoxes) {
    //     cv::rectangle(matchImg, box.roi, cv::Scalar(255,255,0,1), 3);
    // }

    cv::Rect lower_img(0, prevFrame.cameraImg.rows, prevFrame.cameraImg.cols, prevFrame.cameraImg.rows);
    // for (auto box : currFrame.boundingBoxes) {
    //     cv::rectangle(matchImg(lower_img), box.roi, cv::Scalar(0,255,255,1), 3);
    // }

    
    
    // over all roi in prevFrame
    for (auto it=prevFrame.boundingBoxes.begin(); it!=prevFrame.boundingBoxes.end(); ++it) {

        cv::Scalar color = colors[(it-prevFrame.boundingBoxes.begin())%colors.size()];
        cv::rectangle(matchImg, it->roi, color, 3);

        // find all matched keypoints in roi and store index
        vector<vector<int>> bb_match_idx;
        for (auto match : matches) {
            if (it->roi.contains(prevFrame.keypoints[match.queryIdx].pt)) {
                bb_match_idx.push_back({match.queryIdx, match.trainIdx});
            }       
        }

        if (use_graphical_output) {
            for (auto idx : bb_match_idx) {
                //cv::Scalar color = colors[(it-prevFrame.boundingBoxes.begin())%colors.size()];
                cv::circle(matchImg, prevFrame.keypoints[idx[0]].pt, 10, color, 2);  
            }
        }

        // for every keypoint in the roi
        vector<int> corr_count(currFrame.boundingBoxes.size(), 0);
        for (auto idx : bb_match_idx) {
            // for every roi in the currFrame
            //vector<int> corr_count(currFrame.boundingBoxes.size());
            for (auto it2=currFrame.boundingBoxes.begin(); it2!=currFrame.boundingBoxes.end(); ++it2) {
                // roi contains corresponding keypoint
                if (it2->roi.contains(currFrame.keypoints[idx[1]].pt)) {
                    corr_count[it2-currFrame.boundingBoxes.begin()] += 1;
                }
            }
        }

        // find the currFrame roi with the highest number of correspondence counts
        //sort(corr_count.begin(), corr_count.end());
        //currFrame.boundingBoxes[corr_count.back()]
        // for (int i : corr_count) {
        //     cout << i << ", ";
        // }
        // cout << endl;

        int prevFrame_roi_idx = it - prevFrame.boundingBoxes.begin();
        int currFrame_roi_idx = max_element(corr_count.begin(), corr_count.end()) - corr_count.begin();
        
        bbBestMatches[prevFrame_roi_idx] = currFrame_roi_idx;

        if (use_graphical_output) {
            if (accumulate(corr_count.begin(), corr_count.end(), 0) > 100) {
                int roi_idx = max_element(corr_count.begin(), corr_count.end())-corr_count.begin();
                cv::rectangle(matchImg(lower_img), currFrame.boundingBoxes[roi_idx].roi, color, 3);
            }
        }

    }

    for (auto elem : bbBestMatches) {
        cout << "[prev_idx, curr_idx] " << "[" << elem.first << ", " << elem.second << "]\n"; 
    }
    
    
    
    
    

    // for (auto kp : prevFrame.keypoints) {
    //     cv::circle(matchImg, kp.pt, 20, cv::Scalar(255,0,255,1), 2);    
    // }
    // for (auto kp : currFrame.keypoints) {
    //     cv::circle(matchImg(lower_img), kp.pt, 20, cv::Scalar(255,0,255,1), 2);    
    // }

    // for (auto match : matches) {
    //     //cout << "[query, train]: [" << match.queryIdx << ", " << match.trainIdx << "]" << endl; 
    //     //prevFrame.keypoints[match.queryIdx].pt
    //     cv::Point pt2(currFrame.keypoints[match.trainIdx].pt.x, currFrame.keypoints[match.trainIdx].pt.y + prevFrame.cameraImg.rows);
    //     cv::line(matchImg, prevFrame.keypoints[match.queryIdx].pt, pt2, cv::Scalar(255,0,255,1), 2);
    // }

    if (use_graphical_output) {
        string windowName = "Matching keypoints between two camera images";
        cv::namedWindow(windowName, 7);
        cv::resizeWindow(windowName, properties::output_window_width, properties::output_window_height);
        cv::moveWindow(windowName, properties::output_window_pos_x, properties::output_window_pos_y);
        cv::imshow(windowName, matchImg);
        cout << "Press key to continue to next image" << endl;
        cv::waitKey(0); // wait for key to be pressed
    }
}