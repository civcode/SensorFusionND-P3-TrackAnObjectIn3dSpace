#ifndef MY_CV_TOOLS_H_
#define MY_CV_TOOLS_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace mycv {

using namespace cv;

void _prepareImage(InputArray src, const Mat& dst);

void _prepareImgAndDrawKeypoints( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                                         InputArray img2, const std::vector<KeyPoint>& keypoints2,
                                         InputOutputArray _outImg, Mat& outImg1, Mat& outImg2,
                                         const Scalar& singlePointColor, DrawMatchesFlags flags );

void _drawKeypoint( InputOutputArray img, const KeyPoint& p, const Scalar& color, DrawMatchesFlags flags );

void _drawMatch( InputOutputArray outImg, InputOutputArray outImg1, InputOutputArray outImg2 ,
                          const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, DrawMatchesFlags flags );

void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                  InputArray img2, const std::vector<KeyPoint>& keypoints2,
                  const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor,
                  const std::vector<char>& matchesMask, DrawMatchesFlags flags );

}

#endif //MY_CV_TOOLS_H_