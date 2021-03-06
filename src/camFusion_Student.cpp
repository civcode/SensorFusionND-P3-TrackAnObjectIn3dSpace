
#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>
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
        //putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor, 3);
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+80), cv::FONT_ITALIC, 2, currColor, 3);
        sprintf(str2, "xw_min=%2.2fm, yw_mean=%2.2fm", xwmin, ywmax-ywmin);
        //putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor, 3); 
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+155), cv::FONT_ITALIC, 2, currColor, 3);   
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
    cv::resizeWindow(windowName, 800, 800);
    //cv::moveWindow(windowName, 50, 50);
    cv::moveWindow(windowName, 800, 400);
    cv::imshow(windowName, topviewImg);

    //write image for evaluation
    // char file_name[200], str2[200];
    // sprintf(file_name, "../evaluation/lidar_frame_%.2d.png", properties::current_frame_index);
    // cv::imwrite(file_name, topviewImg);

    if(bWait)
    {
        cv::waitKey(10); // wait for key to be pressed
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{ 
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

    // over all roi in prevFrame
    for (auto it=prevFrame.boundingBoxes.begin(); it!=prevFrame.boundingBoxes.end(); ++it) {

        // find all matched keypoints in roi and store index
        vector<vector<int>> bb_match_idx;
        for (auto match : matches) {
            if (it->roi.contains(prevFrame.keypoints[match.queryIdx].pt)) {
                bb_match_idx.push_back({match.queryIdx, match.trainIdx});
            }       
        }

        // for every keypoint in the roi
        vector<int> corr_count(currFrame.boundingBoxes.size(), 0);
        for (auto idx : bb_match_idx) {
            // for every roi in the currFrame
            for (auto it2=currFrame.boundingBoxes.begin(); it2!=currFrame.boundingBoxes.end(); ++it2) {
                // roi contains corresponding keypoint
                if (it2->roi.contains(currFrame.keypoints[idx[1]].pt)) {
                    corr_count[it2-currFrame.boundingBoxes.begin()] += 1;
                }
            }
        }

        int prevFrame_roi_idx = it - prevFrame.boundingBoxes.begin();
        int currFrame_roi_idx = max_element(corr_count.begin(), corr_count.end()) - corr_count.begin();
        
        bbBestMatches[prevFrame_roi_idx] = currFrame_roi_idx;
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    int kn = 100;//50;
    removeLidarOutliers(lidarPointsPrev, properties::reflectivity_min, kn);
    removeLidarOutliers(lidarPointsCurr, properties::reflectivity_min, kn);

    double x_min_prev;
    double x_min_curr;

    x_min_prev = lidarPointsPrev[0].x;
    x_min_curr = lidarPointsCurr[0].x;

    double dT = 1.0/frameRate;

    TTC = x_min_curr * dT / (x_min_prev - x_min_curr);

    cout << "TTC Lidar: " << TTC << endl;
}


void removeLidarOutliers(std::vector<LidarPoint> &lidarPoints, double rmin, int kn) {

    sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint &p1, LidarPoint &p2) -> bool {return p1.x < p2.x;});

    //compute delta_x between two points
    vector<double> dx;
    assert(lidarPoints.size() >= kn+1);
    for (int i=0; i<kn; i++) {
        double d = fabs(lidarPoints[i+1].x-lidarPoints[i].x);
        if (d > numeric_limits<double>::epsilon())
            dx.push_back(d);
    }

    sort(dx.begin(), dx.end());

    double median_dx = dx[dx.size()/2];
    bool is_valid_point = false;
    while (!is_valid_point) {

        if (fabs(lidarPoints[0].x - lidarPoints[1].x) > median_dx *2.0 || lidarPoints[0].r < rmin) {
            lidarPoints.erase(lidarPoints.begin());
        } else {
            is_valid_point = true;
        }
    }
}


void clusterKptMatchesWithROI(BoundingBox &boundingBoxPrev, BoundingBox &boundingBoxCurr, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (auto match : kptMatches) {
        cv::KeyPoint *curr_pt =  &kptsCurr[match.trainIdx];
        cv::KeyPoint *prev_pt =  &kptsPrev[match.queryIdx];

        if (boundingBoxCurr.roi.contains(curr_pt->pt) && 
            boundingBoxPrev.roi.contains(prev_pt->pt) ) { //checking both BB => much better results

            double d = cv::norm(curr_pt->pt - prev_pt->pt);
            if (d > numeric_limits<double>::epsilon()) {
                boundingBoxCurr.kptMatches.push_back(match);
            }
        }
        // store number of kp in roi for evaluation
        if (boundingBoxCurr.roi.contains(curr_pt->pt)) {
            boundingBoxCurr.keypoints.push_back(*curr_pt);
        }
    }
}


void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; 
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { 
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { 

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { 
                // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }     

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }

    // use median to reject outliers
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    
    cout << "TTC Camera: " << TTC << endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera_old(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // 1. randomly pick keypoints and compute distance
    // 2. sort and pick certain number of highest distance pairs
    // 3. compyte quotient
    // 4. compute TTC

    // auto euclidean_distance = [](cv::KeyPoint &p1, cv::KeyPoint &p2) {
    //     return sqrt(pow(p1.pt.x-p2.pt.x,2) + pow(p1.pt.y-p2.pt.y,2));
    // };
    
    typedef struct kp_pair_ {
        cv::DMatch *kp1_match;
        cv::DMatch *kp2_match;
        cv::KeyPoint *kp1;
        cv::KeyPoint *kp2;
        double d;
    } kp_pair;

    // find paris of keypoints in one image and compute the distance
    vector<kp_pair> pairs;
    while (pairs.size() < properties::keypoint_pair_count) {
        cv::DMatch *kp1_match = &kptMatches[rand()%kptMatches.size()];
        cv::DMatch *kp2_match = &kptMatches[rand()%kptMatches.size()];
        //int kp1_idx = kptMatches[rand()%kptMatches.size()].queryIdx;
        //int kp2_idx = kptMatches[rand()%kptMatches.size()].queryIdx;
        //cv::KeyPoint *kp1 = &kptsPrev[kp1_idx];
        //cv::KeyPoint *kp2 = &kptsPrev[kp2_idx];
        cv::KeyPoint *kp1 = &kptsPrev[kp1_match->queryIdx];
        cv::KeyPoint *kp2 = &kptsPrev[kp2_match->queryIdx];

        //double d = euclidean_distance(*kp1, *kp2);
        double d = cv::norm(kp1->pt - kp2->pt);
        //cout << "d: " << d << endl;
        if (d > 1e-2) {
            pairs.push_back({kp1_match, kp2_match, kp1, kp2, d});
        }
    }

    // sort found pairs by distance
    //sort(pairs.begin(), pairs.end(), [](kp_pair p1, kp_pair p2) -> bool {return p1.d>p2.d;});

    // for (auto e : pairs) {
    //     cout << e.d << endl;
    // }

    vector<double> d_quotient;
    for (int i=0; i<properties::pairs_used_count; i++) {
        double d0 = pairs[i].d; //previous distance
        
        //compute current distance
        cv::KeyPoint *kp1 = &kptsCurr[pairs[i].kp1_match->trainIdx];
        cv::KeyPoint *kp2 = &kptsCurr[pairs[i].kp2_match->trainIdx];

        //double d1 = euclidean_distance(*kp1, *kp2);
        double d1 = cv::norm(kp1->pt - kp2->pt);
        //cout << "d0, d1, d1/d0: " << d0 << ", " << d1 << " ," << d1/d0 << endl;
        d_quotient.push_back(d1/d0);

    }

    double median_d = d_quotient[d_quotient.size()/2];
    double mean_d = accumulate(d_quotient.begin(), d_quotient.end(), 0.0) / d_quotient.size();
    double stddev_d = accumulate(d_quotient.begin(), d_quotient.end(), 0.0, 
                                    [mean_d](double val, double d) {
                                        return pow(d-mean_d, 2);
                                    });
    stddev_d = sqrt(stddev_d / d_quotient.size());

    // cout << ">mean_d: " << mean_d << endl;
    // cout << ">stddev_d: " << stddev_d << endl;
    // cout << ">median_d: " << median_d << endl;

    vector<double> d_q;
    for (auto val : d_quotient) {
        if (fabs(val-mean_d) > stddev_d * 2.0) {
            d_q.push_back(val);        
        }
    }

    double d_t = 1.0/frameRate;
    double quotient = accumulate(d_q.begin(), d_q.end(), 0.0) / d_q.size();
    //quotient = mean_d;
    TTC = -d_t / (1 - quotient);

    cout << "TTC: " << TTC << endl;
}


void matchBoundingBoxes_old(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    bool use_graphical_output = false;

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

    // for (auto elem : bbBestMatches) {
    //     cout << "[prev_idx, curr_idx] " << "[" << elem.first << ", " << elem.second << "]\n"; 
    // }
    

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
