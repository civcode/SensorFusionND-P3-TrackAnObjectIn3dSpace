#include "mycvtools.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

//#include "precomp.hpp"

namespace mycv {

using namespace cv;
using namespace std;

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

void _prepareImage(InputArray src, const Mat& dst)
{
    CV_CheckType(src.type(), src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4, "Unsupported source image");
    CV_CheckType(dst.type(), dst.type() == CV_8UC3 || dst.type() == CV_8UC4, "Unsupported destination image");
    const int src_cn = src.channels();
    const int dst_cn = dst.channels();
    
    if (src_cn == dst_cn)
        src.copyTo(dst);
    else if (src_cn == 1)
        cvtColor(src, dst, dst_cn == 3 ? COLOR_GRAY2BGR : COLOR_GRAY2BGRA);
    else if (src_cn == 3 && dst_cn == 4)
        cvtColor(src, dst, COLOR_BGR2BGRA);
    else if (src_cn == 4 && dst_cn == 3)
        cvtColor(src, dst, COLOR_BGRA2BGR);
    else
        CV_Error(Error::StsInternal, "");
    
}

void _prepareImgAndDrawKeypoints( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                                         InputArray img2, const std::vector<KeyPoint>& keypoints2,
                                         InputOutputArray _outImg, Mat& outImg1, Mat& outImg2,
                                         const Scalar& singlePointColor, DrawMatchesFlags flags )
{
    Mat outImg;
    Size img1size = img1.size(), img2size = img2.size();
    //Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height) );
    Size size( MAX(img1size.width, img2size.width), img1size.height + img2size.height );
    
    if( !!(flags & DrawMatchesFlags::DRAW_OVER_OUTIMG) )
    {
        outImg = _outImg.getMat();
        if( size.width > outImg.cols || size.height > outImg.rows )
            CV_Error( Error::StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
        // outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) );
        // outImg2 = outImg( Rect(img1size.width, 0, img2size.width, img2size.height) );
        
        outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) );
        outImg2 = outImg( Rect(0, img1size.height, img2size.width, img2size.height) );
    }
    else
    {
        const int cn1 = img1.channels(), cn2 = img2.channels();
        const int out_cn = std::max(3, std::max(cn1, cn2));
        _outImg.create(size, CV_MAKETYPE(img1.depth(), out_cn));
        outImg = _outImg.getMat();
        outImg = Scalar::all(0);
        // outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) );
        // outImg2 = outImg( Rect(img1size.width, 0, img2size.width, img2size.height) );
        
        outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) );
        outImg2 = outImg( Rect(0, img1size.height, img2size.width, img2size.height) );

        _prepareImage(img1, outImg1);
        _prepareImage(img2, outImg2);
    }
    
    // draw keypoints
    if( !(flags & DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) )
    {
        Mat _outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) );
        drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor, flags | DrawMatchesFlags::DRAW_OVER_OUTIMG );

        Mat _outImg2 = outImg( Rect(0, img1size.height, img2size.width, img2size.height) );
        drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor, flags | DrawMatchesFlags::DRAW_OVER_OUTIMG );
    }
}

void _drawKeypoint( InputOutputArray img, const KeyPoint& p, const Scalar& color, DrawMatchesFlags flags )
{
    CV_Assert( !img.empty() );
    Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );

    if( !!(flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS) )
    {
        int radius = cvRound(p.size/2 * draw_multiplier); // KeyPoint::size is a diameter

        // draw the circles around keypoints with the keypoints size
        circle( img, center, radius, color, 1, LINE_AA, draw_shift_bits );

        // draw orientation of the keypoint, if it is applicable
        if( p.angle != -1 )
        {
            float srcAngleRad = p.angle*(float)CV_PI/180.f;
            Point orient( cvRound(cos(srcAngleRad)*radius ),
                          cvRound(sin(srcAngleRad)*radius )
                        );
            line( img, center, center+orient, color, 1, LINE_AA, draw_shift_bits );
        }
#if 0
        else
        {
            // draw center with R=1
            int radius = 1 * draw_multiplier;
            circle( img, center, radius, color, 1, LINE_AA, draw_shift_bits );
        }
#endif
    }
    else
    {
        // draw center with R=3
        int radius = 3 * draw_multiplier;
        circle( img, center, radius, color, 1, LINE_AA, draw_shift_bits );
    }
}

void _drawMatch( InputOutputArray outImg, InputOutputArray outImg1, InputOutputArray outImg2 ,
                          const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, DrawMatchesFlags flags )
{
    RNG& rng = theRNG();
    bool isRandMatchColor = matchColor == Scalar::all(-1);
    Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256), 255 ) : matchColor;

    _drawKeypoint( outImg1, kp1, color, flags );
    _drawKeypoint( outImg2, kp2, color, flags );

    Point2f pt1 = kp1.pt,
            pt2 = kp2.pt,
            //dpt2 = Point2f( std::min(pt2.x+outImg1.size().width, float(outImg.size().width-1)), pt2.y )
            dpt2 = Point2f( pt2.x, pt2.y+outImg1.size().height )
            ;

    line( outImg,
          //Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
          //Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
          Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
          Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
          color, 1, LINE_AA, draw_shift_bits );
}

void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                  InputArray img2, const std::vector<KeyPoint>& keypoints2,
                  const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                  const Scalar& matchColor, const Scalar& singlePointColor,
                  const std::vector<char>& matchesMask, DrawMatchesFlags flags )
{
    if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
        CV_Error( Error::StsBadSize, "matchesMask must have the same size as matches1to2" );

    Mat outImg1, outImg2;
    _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
                                 outImg, outImg1, outImg2, singlePointColor, flags );

    // draw matches
    for( size_t m = 0; m < matches1to2.size(); m++ )
    {
        if( matchesMask.empty() || matchesMask[m] )
        {
            int i1 = matches1to2[m].queryIdx;
            int i2 = matches1to2[m].trainIdx;
            CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));

            const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
            _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
        }
    }
}

} // namespace mycv



