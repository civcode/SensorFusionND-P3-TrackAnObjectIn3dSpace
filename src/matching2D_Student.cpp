#include <numeric>
#include "matching2D.hpp"
#include "properties.h"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = properties::use_matcher_cross_check;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t = (double)cv::getTickCount();

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if (properties::feature_descriptor_type.compare("SIFT") == 0 || 
            properties::feature_descriptor_type.compare("SURF") == 0) {
            normType = cv::NORM_L2;
        } else { 
            normType = cv::NORM_HAMMING;
        }
        //normType = cv::NORM_L2;
        //normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        double dist_ratio_min = 0.8;
        for (auto it=knn_matches.begin(); it!=knn_matches.end(); ++it) {
            if ((*it)[0].distance < dist_ratio_min * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }
        
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << matcherType << " matching n= " << matches.size() << " features in " << 1000 * t / 1.0 << " ms" << endl;
    properties::frame_data[properties::current_frame_index].matcher_time = t;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool use_orientation = false;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int orb_nfeatures = 3000;
        float orb_scale_factor = 1.2f;
        int orb_nlevels = 8;

        extractor = cv::ORB::create(orb_nfeatures, orb_scale_factor, orb_nlevels);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientatin_normalized = true;
        bool scale_normalized = true;
        float pattern_scale = 22.0f;
        int n_octaves = 4;

        extractor = cv::xfeatures2d::FREAK::create(orientatin_normalized, scale_normalized, pattern_scale, n_octaves);

    }else if (descriptorType.compare("AKAZE") == 0)
    {
        // AKAZE descriptors can only be used with AKAZE keypoints 
        if (properties::keypoint_detector_type.compare("AKAZE") != 0) {
            cout << "error: AKAZE descriptors can only be used with AKAZE keypoits. you are using " << properties::keypoint_detector_type << " keypoints." << endl;
            exit(-1);
        }
        extractor = cv::AKAZE::create();

    }else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();    

    }else if (descriptorType.compare("SURF") == 0)
    {  
        extractor = cv::xfeatures2d::SURF::create();  
    }


    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    properties::frame_data[properties::current_frame_index].descriptor_time = t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    properties::frame_data[properties::current_frame_index].detector_time = t;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) 
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; //100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm; //, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // add corners to result vector
    for (int row=0; row<dst_norm.rows; row++) {
        for (int col=0; col<dst_norm.cols; col++) {
            float response = dst_norm.at<float>(row, col);
            if (response > minResponse) {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(col, row);
                newKeyPoint.response = response;
                newKeyPoint.size = 2*apertureSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    properties::frame_data[properties::current_frame_index].detector_time = t;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) 
{
    cv::Ptr<cv::FeatureDetector> detector = nullptr;

    if (detectorType.compare("FAST") == 0)  {
        int fast_threshold = 30;
        bool fast_nms = true;

        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(fast_threshold, fast_nms, type);
    
    } else if (detectorType.compare("BRISK") == 0)  {
        int brisk_threshold = 30;
        int brisk_octaves = 3;
        float brisk_pattern_scale = 1.0f;

        detector = cv::BRISK::create(brisk_threshold, brisk_octaves, brisk_pattern_scale);
    
    } else if (detectorType.compare("ORB") == 0)  {
        int orb_nfeatures = 3000;
        float orb_scale_factor = 1.2f;
        int orb_nlevels = 8;

        detector = cv::ORB::create(orb_nfeatures, orb_scale_factor, orb_nlevels);
    
    } else if (detectorType.compare("SIFT") == 0)  {

        detector = cv::xfeatures2d::SIFT::create();
    
    } else if (detectorType.compare("SURF") == 0)  {

        detector = cv::xfeatures2d::SURF::create();
    
    } else if (detectorType.compare("AKAZE") == 0)  {

        detector = cv::AKAZE::create();
    }
    
    // run detector
    if (detector != nullptr) {
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        properties::frame_data[properties::current_frame_index].detector_time = t;
    }

    


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}