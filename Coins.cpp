
#define _USE_MATH_DEFINES
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "HiResTimer.h";
#include <queue>
#include <iomanip>
#include<utility>
#include <algorithm>    // std::nth_element
//#include "cvBlob\cvblob.h"
#include <tesseract/baseapi.h>


using namespace cv;
using namespace std;

/// Global variables

Mat src, src_gray, src_gray_rotated;
Mat preprocessedImage, detected_edges, dateCroppedImage;
Mat mask;
Mat ocrImage;
Mat contoursImg;

int const imageWidth = 500;
int imageHeight;
int edgeThresh = 1;
int bluringFactor = 15;
int cannyLowThreshold = 16;
int detectPressed;
int const max_lowThreshold = 100;
int const maxBluringFactor = 99;

int preRotationAngle = 0;

int ratio = 3;
int kernel_size = 3;
char* canny_window = "Edge Map";
char* bluringWindow = "Bluring";
char* contoursWindow = "Contours";
char* resultWindow = "Date";

int readDateArgs[2];

//Date Contours Properties
int dateMinHeight;
int dateMaxHeight;
int dateMinWidth;
int dateMaxWidth;
float dateMinAspectRatio;
float dateMaxAspectRatio;

int dateMinContoursPoints;
float dateMinExtent;
float dateMinSolidity;
float dateMinSolidity2;
float dateMaxDeviationAngle;
float dateMaxRelativeAngle;
float dateMinRelativeAngle;

int dateMinContoursCount;
int contourRecMaxRatio;
int maxContourLength;
int minContourLength;
int noizeCountourWidth;
int digitsDistance;


std::vector<std::vector<Point> > contours;
std::vector<Vec4i> hierarchy;


/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/

RNG rng(12345);

enum PROCESSING_STAGE{
	STAGE_PREPROCESS_IMAGE,
	STAGE_EXTRACT_COIN,
	STAGE_CANNY_THRESHOLD,
	STAGE_EXTRACT_DATE_RECT,
	STAGE_PREPROCESS_DATE_IMAGE,
	STAGE_OCR_DATE
}processingStage;

void extractDate(int, void*);

void CannyThreshold(int, void*)
{
	static int last_cannyLowThreshold = -1;
	if(last_cannyLowThreshold == cannyLowThreshold)
		return;
	last_cannyLowThreshold = cannyLowThreshold;
	if(!preprocessedImage.data)
		return;
	
	//// for image watcher
	//Mat b = preprocessedImage;
	//Mat de = detected_edges;
	
	/// Canny detector
	Canny(preprocessedImage, detected_edges, cannyLowThreshold, cannyLowThreshold*ratio, kernel_size, true);

	if(!mask.data){
		// create mask to select region of interrest
		mask.create(detected_edges.size(), detected_edges.type());  // create an Mat that has same Dimensons as src
		mask.setTo(cv::Scalar(0));                                            // creates
		cv::Point center(imageWidth/2, imageHeight/2); // CVRound converts floating numbers to integer
		int largeRadius = imageWidth * 0.45, smallRadius = imageWidth * 0.22;                              // Radius is the third parameter [i][0] = x [i][1]= y [i][2] = radius
		cv::circle( mask, center, largeRadius, cv::Scalar(255),-1, 8, 0 );
		cv::circle( mask, center, smallRadius, cv::Scalar(0),-1, 8, 0 );    // Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)

	}
	//GaussianBlur(detected_edges, detected_edges,Size(3,3), 3,3);
	cv::morphologyEx( detected_edges, detected_edges, cv::MORPH_CLOSE, Mat());
	//dilate(detected_edges, detected_edges,Mat());

	Mat masked;
	detected_edges.copyTo(masked,mask); // creates masked Image and copies it to maskedImage
	detected_edges = masked;
	
}

inline Rect getLargestCircleRect(Mat img){
	if(!img.data)
		return Rect();
	
	Mat tmp;
	
	/// Convert the image to grayscale
	cvtColor(img, tmp, CV_BGR2GRAY);

		

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( tmp, tmp, Size(9, 9), 3,3 );

	vector<Vec3f> circles;
	HoughCircles(tmp, circles, CV_HOUGH_GRADIENT, 2, 
		img.cols/3, 100, 200, img.cols/6, img.cols/2);

	  /// Draw the circles detected
	int r = 0;
	Point c;
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		
		if(r < radius){
			c = center;
			r = radius;
		}

		//// circle center
		//circle( src_gray, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		//circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
	if(circles.size() > 0){
		// return largest circle rect
		int x = c.x-r, y = c.y-r, w = r * 2;
		//range check, if((unsigned)(number-lower) <= (upper-lower) ) inRange(number)
		if ((unsigned)x <= img.cols &&
			(unsigned)y <= img.rows &&
			(unsigned)(x+w) <= img.cols &&
			(unsigned)(y+w) <= img.rows )
		return Rect(x, y, w,w);
	}
	return Rect();
}

void preProcessImage(int, void*)
{
	static int lastBluringFactor = 0;
	if(lastBluringFactor == bluringFactor)
		return;
	if(!src.data)
		return;
//	Mat b = preprocessedImage; // for image watcher
	
	Rect coinRect = getLargestCircleRect(src);
	src = src(coinRect);	

	//rectangle(src, coinRect, Scalar(255,255,0),2);	

	// resize
	imageHeight = imageWidth * src.rows / src.cols;
	cv::resize(src, src, cv::Size(imageWidth, imageHeight));

	
	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Reduce noise with a kernel 3x3
	bluringFactor |= 0x1; // make it odd number
	//blur(src_gray, preprocessedImage, Size(bluringFactor,bluringFactor));
	//medianBlur(src_gray, preprocessedImage, bluringFactor); 
	//bilateralFilter ( src_gray, preprocessedImage, bluringFactor, bluringFactor*2, bluringFactor/2 );
	GaussianBlur(src_gray, preprocessedImage, cv::Size(bluringFactor,bluringFactor), 3, 0);


}


inline float p2pDistance(const Point2f &a,const Point2f &b){
	float d = sqrt(	pow(abs(a.x - b.x), 2) +  pow(abs(a.y - b.y), 2) );
	//double d = MAX(abs(a.x - b.x), abs(a.y - b.y) );
	return d ;
}
inline float r2rDistance(const RotatedRect &a, const RotatedRect &b){
	float d;

	//d = p2pDistance(a.center, b.center);
	//d -= ( (a.size.width, a.size.height) /4);
	//d -= ( (b.size.width, b.size.height) /4);
	//return d;

	Point2f aPoints[4];
	a.points(aPoints);
	Point2f bPoints[4];
	b.points(bPoints);
	float minD = imageWidth;
	for(Point2f& ap : aPoints){
		for(Point2f& bp : bPoints){
			d = p2pDistance(ap, bp);
			if(d < minD)
				minD = d;
		}

	}
	return minD;
}
inline Point massCenter(const std::vector<Point>& points){
	int s = points.size();
	LONGLONG x = 0, y = 0;
	for(const Point &p: points){
		x += p.x ;
		y += p.y;
	}
	x /= s;
	y/= s;
	return Point(x,y);
}
inline float cHull2HullcDistance(const vector<Point> &a,const vector<Point> &b){
	std::vector<Point > h1, h2;
	convexHull(a, h1);
	convexHull(b, h2);
	float d, minD = imageWidth;
	for(const Point &p1 : h1)
		for(const Point &p2 : h2){
			d = p2pDistance(p1, p2);
			if(d < minD)
				minD = d;
		}
		return minD;
}
inline float c2cDistance(const vector<Point> &a,const vector<Point> &b){

	float d, minD = imageWidth;
	for(const Point &p1 : a)
		for(const Point &p2 : b){
			d = p2pDistance(p1, p2);
			if(d < minD)
				minD = d;
		}
		return minD;

	//Point2f c1, c2;
	//float r1, r2, d;
	//c1 = massCenter(a);
	//c2 = massCenter(b);
	//d = p2pDistance(c1, c2);
	//
	//return d;
}

inline float c2cRelativeDistance(const vector<Point> &a,const vector<Point> &b){
	float aArea = contourArea(a);
	float bArea = contourArea(b);
	float value = sqrt(aArea) + sqrt(bArea);
	float d = c2cDistance(a,b);
	d /= value;
	return d;
}
typedef pair<int, int> P;
class KeyValueComparison
{
  bool reverse;
public:
  KeyValueComparison(const bool& revparam=false)
    {reverse=revparam;}
  bool operator() (const P& lhs, const P&rhs) const
  {
	   return (lhs.second>rhs.second);
  }
};
//typedef std::priority_queue<P, std::vector<P>,KeyValueComparison> KeyValue_priority_queue;
std::string ocrNumber(int, void*){
	if(!ocrImage.data)
		return "";
	
	tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
    tess.SetVariable("tessedit_char_whitelist", "0l23456789"); // 1 replaced with l char to work around '1' proplem
	tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
    tess.SetImage((uchar*)ocrImage.data, ocrImage.cols, ocrImage.rows, 1, ocrImage.cols);
	std::string strResult = tess.GetUTF8Text();
	// replace l letter with 1 digit
	std::replace(strResult.begin(), strResult.end(), 'l', '1');
	//std::cout << "OCR recognition time: "<< timer.GetElapsedMilliSeconds() << std::endl;
	
	if(!strResult.empty())
		cout << "Detected date: " << strResult << endl;
	//else
		//cout << "OCR engine can't recognize date\n";
	return strResult;
}

void preprocessDate(int, void*){

	if(!dateCroppedImage.data)
		return;

			// OCR extracted image
			//char* strTest = "2010";
			//int bl;
			//Size s = getTextSize(strTest, 0, 1, 3,&bl );
			//Mat test( s, CV_8UC1,Scalar(0) );
			//putText(test, strTest, Point(0,s.height),0,1,Scalar(255), 3); 



			GaussianBlur(dateCroppedImage, ocrImage, cv::Size(7,7), 3, 0);
			//threshold(ocrImage, ocrImage,0,255,THRESH_BINARY_INV|THRESH_OTSU);
			adaptiveThreshold(ocrImage, ocrImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, readDateArgs[0]|0x1,readDateArgs[1]); 
			//cv::morphologyEx( ocrImage, ocrImage, cv::MORPH_ERODE, Mat(),Point(-1,-1), 1);
			//cv::morphologyEx( ocrImage, ocrImage, cv::MORPH_CLOSE, Mat(),Point(-1,-1), 2);
			//cv::morphologyEx( ocrImage, ocrImage, cv::MORPH_DILATE, Mat(),Point(-1,-1), 1);
			

			//Mat dateEdges;
			//float threshold = 9;
			//Canny(ocrImage, dateEdges, threshold, threshold*ratio, kernel_size);
			//cv::morphologyEx( dateEdges, dateEdges, cv::MORPH_CLOSE, Mat());
			//
			////GaussianBlur(dateCroppedImage, ocrImage, cv::Size(9,9), 3, 0);
			//ocrImage = dateEdges;

			//std::vector<std::vector<cv::Point> > croppedContours;
			//vector<Vec4i> hierarchy;
			//findContours( ocrImage, croppedContours, hierarchy,
			//	CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
			//ocrImage = Mat( dateCroppedImage.size(), CV_8UC1, Scalar(0) );
			//CvMat ocrImgC(ocrImage);
			//int idx;
			////for( ; idx >= 0; idx = hierarchy[idx][0] )
			//for(idx = 0;idx < croppedContours.size(); idx++)
			//{
			//	if(croppedContours[idx].size() < 10)
			//		continue;
			//	if (  hierarchy[idx][3] == -1) // has no  parent, noize 
			//		  continue;
			//	if( hierarchy[hierarchy[idx][3]][3] != -1)  // has gradpa (hole)
			//	{ 
			//		//if(croppedContours[idx].size() < 30)
			//			//continue;
			//		// delete filled by parent
			//		drawContours(ocrImage, croppedContours, idx, Scalar(0), -1,8, hierarchy, 0);
			//	}
			//	else
			//		drawContours(ocrImage, croppedContours, idx, Scalar(255), -1,8, hierarchy, 0);//fill

			//}

			
			// calculate mean of date rect
			//Scalar mean = cv::mean(dateCroppedImage);

			// save to filtered rects
			//tmp.push_back(rects[i]);


}
void extractDate(int, void*){
	if(!detected_edges.data || 
		!src_gray.data)
		return;
	Scalar white(255, 255, 255);
	Scalar red(0, 0, 255);
	Scalar blue(255,0,0);
	Scalar green(0,255,0);
	contoursImg = Mat::zeros( detected_edges.size(), CV_8UC3 );

	Mat edges = detected_edges;

	/// Find contours
	findContours( detected_edges, contours,
		CV_RETR_LIST,   CV_CHAIN_APPROX_SIMPLE );
	

	// get  bounding rects
    std::vector<RotatedRect> rects;
    std::vector<std::vector<Point> > filteredContours;
	std::vector< std::vector< int > > contoursOrigin;
	for( int i = 0; i< contours.size(); i++ )
	{
		RotatedRect minRect = minAreaRect(contours[i]);
		int largeSide = MAX(minRect.size.width, minRect.size.height);
		int smallSide = MIN(minRect.size.width, minRect.size.height);
		if(		largeSide < dateMaxWidth &&
				smallSide < dateMaxHeight &&
				largeSide > noizeCountourWidth 
			)
		{
			
			rects.push_back(minRect);
			filteredContours.push_back(contours[i]);
			vector<int> tmp(1);tmp[0] = i;
			contoursOrigin.push_back(tmp); 
			/// Draw contours
			drawContours( contoursImg, contours, i, white  );
		}
	}
	bool changed;
	do{
		changed = false;
        std::vector<bool> deleted( rects.size(),false);
		for(int i = 0; i < rects.size(); i++){
			if(deleted[i])
				continue;
            priority_queue<P, std::vector<P>,KeyValueComparison> neighbors;
			for(int j = i+1; j < rects.size(); j++){
				if(deleted[j])
					continue;
				if(p2pDistance( rects[i].center, rects[j].center) > dateMaxWidth)
					continue; // too far 
				// ToDo: check if one contains other
				
				float d = p2pDistance(rects[i].center, rects[j].center);
				if(d > dateMaxWidth)
					continue;
				
				d = c2cRelativeDistance(filteredContours[i], filteredContours[j]);
				if(d < (float)digitsDistance/100.0){
					neighbors.push(pair<int, int>(j, d));
				}
			}
			while(!neighbors.empty()){
				pair<int,int> p = neighbors.top();
				neighbors.pop();
                std::vector<Point> tmp = filteredContours[i];
				tmp.insert( tmp.end(), filteredContours[p.first].begin(), filteredContours[p.first].end()); 
				RotatedRect minRect = minAreaRect(tmp);

				int largeSide = MAX(minRect.size.width, minRect.size.height);
				int smallSide = MIN(minRect.size.width, minRect.size.height);
				if(		largeSide < dateMaxWidth &&
						smallSide < dateMaxHeight
					)
				{
					filteredContours[i] = tmp;
					contoursOrigin[i].insert(contoursOrigin[i].end(), contoursOrigin[p.first].begin(), contoursOrigin[p.first].end() );
					rects[i] = minRect;
					deleted[p.first] = true;
					changed = true;
				}
			}
		}
        std::vector<RotatedRect> tmpRects;
		std::vector<std::vector<Point> > tmpC;
		std::vector<std::vector<int> >tmpO;
		for(int i = 0; i < rects.size(); i++)
			if(!deleted[i]){
				tmpRects.push_back(rects[i]);
				tmpC.push_back(filteredContours[i]);
				tmpO.push_back(contoursOrigin[i]);
			}
		rects = tmpRects;
		filteredContours = tmpC;
		contoursOrigin = tmpO;
	}while(changed);

	for( int i = 0; i< rects.size(); i++ )
	{
		float largeSide = MAX(rects[i].size.width, rects[i].size.height);
		float smallSide = MIN(rects[i].size.width, rects[i].size.height);
		float aspectRatio = smallSide / largeSide;
		
		// check window range and some features
		if(		smallSide < dateMinHeight 
				 || smallSide > dateMaxHeight 
				 || largeSide < dateMinWidth 
				 || largeSide > dateMaxWidth 
				|| aspectRatio < dateMinAspectRatio
				|| aspectRatio > dateMaxAspectRatio
				|| contoursOrigin[i].size() < dateMinContoursCount
			)
			continue;
		
		float cArea = 0;
		std::vector<Point > hull;
		float rArea = largeSide * smallSide;
		float hull_area = 0;

		convexHull(filteredContours[i], hull);
		hull_area += contourArea(hull);
//			Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		for(const int &originIndex :  contoursOrigin[i]){
			cArea += contourArea(contours[originIndex]);
			drawContours(contoursImg, contours, originIndex, red); 
		}
		float solidity = cArea/hull_area;
		float solidity2 = hull_area / rArea;
		float extent = cArea/rArea;
		int cPoints = filteredContours[i].size();
		double l = arcLength(filteredContours[i], false);
		//Point2f mc = massCenter(filteredContours[i]);
		Vec4f mLine;
		fitLine( filteredContours[i], mLine, CV_DIST_L12, 0, 0.01, 0.01); 
		float lAngle = atan2(mLine[1], mLine[0]) * 180.0 / M_PI ;
		float rAngle = rects[i].angle;
		float centeralAngle = atan2(	rects[i].center.y - imageHeight/2,	// arctan(y2-y1 / x2-x1)
										rects[i].center.x - imageWidth/2)
										* 180.0 / M_PI ;
		if(rects[i].size.width < rects[i].size.height){
			rAngle += 90;
		}
		lAngle += 360;
		rAngle += 360;
		centeralAngle += 360;
		lAngle = fmod(lAngle,360);
		rAngle = fmod(rAngle,360);
		centeralAngle = fmod(centeralAngle,360);
		float deviation = abs(lAngle - rAngle);
		float relativeAngle = centeralAngle - rAngle;
		relativeAngle += 360;
		relativeAngle = fmod(relativeAngle, 180);

		if(
				////contours features
				cPoints < dateMinContoursPoints 
				|| extent < dateMinExtent
				|| solidity < dateMinSolidity
				|| solidity2 <  dateMinSolidity2
				|| deviation > dateMaxDeviationAngle
				|| relativeAngle < dateMinRelativeAngle
				|| relativeAngle > dateMaxRelativeAngle
			)
			continue; // skip not matched

	// if date located out of its location, flip it
		if( rects[i].center.x < imageWidth/2 ) // left side
		{
			// top left quarter always upside down
			if(rects[i].center.y < imageHeight/2){ 
				rAngle += 180;
				rAngle = fmod(rAngle,360);
			}
			// Top left quarter flipps, check rotation angle
			else{
				if(rAngle > 90){
					rAngle += 180;
					rAngle = fmod(rAngle,360);
				}
				// else, on right orientation
			}
		}
		else{ // right side
			if(rects[i].center.y < imageHeight/2) //top right quarter flipps, check rotation angle
			{
				if(rAngle <= 90){
					rAngle += 180;
					rAngle = fmod(rAngle,360);					
				}
				// else, on right orientation
			}
			// else,  bottom right quarter always right 
		}
		// extract and rotate date rectangle

		// get the rotation matrix
		Mat M = getRotationMatrix2D(rects[i].center, rAngle, 1.0);
			
		////----------------------------------------------------------------------------
		////---------------------- Draw date contours transformed (rotated)-------------
		////----------------------------------------------------------------------------
		// Mat dateCroppedImage = Mat::zeros( detected_edges.size(), CV_8UC1 );
		//for(const int &originIndex :  contoursOrigin[i]){
		//		cv::transform(contours[originIndex], contours[originIndex], M);
		//		drawContours(dateCroppedImage, contours, originIndex, Scalar(255), -1);
		//}
		//int x = rects[i].center.x - largeSide /2;
		//int y = rects[i].center.y - smallSide /2;
		//Rect rotatedRect(x, y, largeSide, smallSide);
		//dateCroppedImage = dateCroppedImage(rotatedRect);
		////-------------------------------------- OR -----------------------------------
		////------------------------ extract date from source image ----------------
		////-----------------------------------------------------------------------------
		dateCroppedImage = Mat::zeros( src_gray.size(), src_gray.type() );
		// perform the affine transformation
		warpAffine(src_gray, dateCroppedImage, M, src_gray.size(), CV_INTER_LINEAR);
		// crop the resulting image
		getRectSubPix(dateCroppedImage, Size(largeSide+8, smallSide+8), rects[i].center, dateCroppedImage);
		////==============================================================================

			
			//draw red rectangle on date 
		Point2f rect_points[4]; rects[i].points(rect_points);
		for( int j = 0; j < 4; j++ )
			line( contoursImg, rect_points[j], rect_points[(j+1)%4], green, 2, 8 );
				
		//// Draw contour property value
		//std::ostringstream ss;
		//ss << std::setprecision(0)   << (smallSide / largeSide) ; 
		//Point p = rects[i].center;
		//p.x -= 50;
		//p.y -= 0;
		//cv::putText(contoursImg, ss.str(), p, 0, 0.6, green, 2, 8); 
		//ss = std::ostringstream();
		//ss << std::setprecision(0)   << "rAngle: " << rAngle ; 
		//p.y += 20;
		//cv::putText(contoursImg, ss.str(), p, 0, 0.6, red, 2, 8); 
		//ss = std::ostringstream();
		//ss << std::setprecision(0)   <<  rects[0].angle ; 
		//p.y += 20;
		//cv::putText(contoursImg, ss.str(), p, 0, 0.6, Scalar(0,255,255), 2, 8); 
		//		
		//// draw mean line
		//int lefty = int((- mLine[2] *mLine[1]/mLine[0]) + mLine[3]);
		//int righty = int(((imageWidth-mLine[2])*mLine[1]/mLine[0])+mLine[3]);
		//cv::line(contoursImg,  Point(imageWidth-1,righty), Point(0,lefty),blue,1);	

		//draw center line
		//cv::line(contoursImg, Point(imageWidth/2, imageHeight/2), rects[i].center, blue, 2);

		// draw center and mass center
		//circle(contoursImg, mc, 5, blue, CV_FILLED);  
		//circle(contoursImg, rects[i].center, 5, red, CV_FILLED);  
		//

		break; // just first detected date 
	}
}
void processImage(int, void* param){
	
	CHiResTimer timer;
	timer.Init();
	
	// pre rotate
	//Mat M = getRotationMatrix2D(Point(imageWidth/2, imageHeight/2), preRotationAngle, 1.0);
	//warpAffine(src_gray, src_gray_rotated, M, src_gray.size(), INTER_NEAREST);
	PROCESSING_STAGE stage = STAGE_PREPROCESS_IMAGE, *pStage = (PROCESSING_STAGE*)param;
	if (pStage != NULL) stage = *pStage;
	
	if(stage <=  STAGE_PREPROCESS_IMAGE){
		
		preProcessImage(0,0);

		//int msec = timer.GetElapsedMilliSeconds();
		//std::cout << "Bluring time: " << msec << " msec\n";
		imshow(bluringWindow, preprocessedImage);
	}
	
	if(stage <= STAGE_CANNY_THRESHOLD){
		CannyThreshold(0,0);
		imshow(canny_window, detected_edges);
	
		//int msec = timer.GetElapsedMilliSeconds();
		//std::cout << "Edge detecting time: " << msec << " msec\n";

	}
	
	if(stage <= STAGE_EXTRACT_DATE_RECT){
		extractDate(0,0);
		imshow(contoursWindow, contoursImg);
	}
	
	if(stage <= STAGE_PREPROCESS_DATE_IMAGE){
		preprocessDate(0,0);
		if(ocrImage.data)
			imshow(resultWindow, ocrImage);
	}
	if(stage <= STAGE_OCR_DATE){
		ocrNumber(0,0);
	}
	
	cout << "Processing time: " << timer.GetElapsedMilliSeconds() << " msec.\n";
	imshow("Original", src);
}

/** @function main */

inline void setThresholds(){
	
	dateMinWidth = imageWidth * 0.16;
	dateMinHeight = imageWidth * 0.058;
	dateMaxWidth = imageWidth * 0.204;
	dateMaxHeight = imageWidth * 0.11;
	
	dateMinAspectRatio = 0.3;
	dateMaxAspectRatio = 0.45;
		
	dateMinExtent = 0.3;
	dateMinSolidity = 0.2;
	dateMinSolidity2 = 0.7;
	dateMinContoursPoints = 270;
	dateMinContoursCount = 5;
	dateMaxDeviationAngle = 10.0;
	dateMinRelativeAngle = 15.0;
	dateMaxRelativeAngle = 45.0;
	
	dateMinContoursPoints = 100;

	digitsDistance = 70; // relative to digit size
	noizeCountourWidth = imageWidth * 0.03;
	contourRecMaxRatio = 5;
	maxContourLength = imageWidth * 0.5;
	minContourLength = imageWidth * 0.03;
	
	readDateArgs[0] = 19;
	readDateArgs[1] = 9;

}

int main(int argc, char** argv)
{
	if(argc < 2)
		return -1;

	/// Load an image
	src = imread(argv[1]);

	if (!src.data)
	{
		return -1;
	}
	setThresholds();

	//
	///// Create a window
	namedWindow(bluringWindow, CV_WINDOW_AUTOSIZE);
	//
	///// Create a Trackbar for user to enter threshold
	//createTrackbar("Bluring factor:", bluringWindow, &bluringFactor, maxBluringFactor, preProcessImage);
	///// Create a Trackbar for user to enter threshold
	//createTrackbar("Rotation Angle:", bluringWindow, &preRotationAngle, 720, preRotate);

	/// Create a window
	//namedWindow(canny_window, CV_WINDOW_AUTOSIZE);

	///// Create a Trackbar for user to enter threshold
	//createTrackbar("Min Threshold:", canny_window, &cannyLowThreshold, max_lowThreshold, CannyThreshold);

	char* controlsWindow = "Date Contours Properties";
	/// Create a window
	namedWindow(controlsWindow, CV_WINDOW_FREERATIO);
	cv::resizeWindow(controlsWindow, 400, 500);
	
	PROCESSING_STAGE extractDate = STAGE_EXTRACT_DATE_RECT;

	/// Create a Trackbars for user to enter threshold
	createTrackbar("Min Height:",				controlsWindow, &dateMinHeight, imageWidth, processImage, &extractDate);
	createTrackbar("Max Height:",				controlsWindow, &dateMaxHeight, imageWidth, processImage, &extractDate);
	createTrackbar("Min Width:",				controlsWindow, &dateMinWidth, imageWidth, processImage, &extractDate);
	createTrackbar("Max Width:",				controlsWindow, &dateMaxWidth, imageWidth, processImage, &extractDate);
	createTrackbar("Digits Splitting Distance:",controlsWindow, &digitsDistance, 100, processImage, &extractDate);
	
	/////// Create a window
	//namedWindow(contoursWindow, CV_WINDOW_AUTOSIZE);
	
	///// Create a window
	namedWindow(resultWindow, CV_WND_PROP_ASPECTRATIO);
	PROCESSING_STAGE dateImageStage = STAGE_PREPROCESS_DATE_IMAGE;
	createTrackbar("arg1:", resultWindow, readDateArgs , 100, processImage, (void*)&dateImageStage);
	createTrackbar("arg2:", resultWindow, readDateArgs+1 , 100, processImage, (void*)&dateImageStage);


	processImage(0,0);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}
