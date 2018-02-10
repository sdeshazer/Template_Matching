

//opencv2:
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
//opencv:
#include <opencv/highgui.h>
#include <opencv/cv.h>
//====================
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

// ==[namespace]==
using namespace std;


// constants:
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 640;
const int MAX_NUM_OBJECTS = 40;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//=========================================================

// ================
//   intToString
// ================
std::string intToString(int value) {
	std::stringstream ss;
	ss << value;
	return ss.str();

}//inToString

 // ================
 //   drawObject
 // ================
void drawObject(int x, int y, cv::Mat &frame) {
	//crosshairs:
	circle(frame, cv::Point(x, y), 20, cv::Scalar(0, 255, 0), 2);

	if (y - 25 > 0)
		cv::line(frame, cv::Point(x, y), cv::Point(x, y - 25), cv::Scalar(0, 255, 0), 2);
	else cv::line(frame, cv::Point(x, y), cv::Point(x, 0), cv::Scalar(0, 255, 0), 2);

	if (y + 25 < FRAME_HEIGHT)
		cv::line(frame, cv::Point(x, y), cv::Point(x, y + 25), cv::Scalar(0, 255, 0), 2);
	else cv::line(frame, cv::Point(x, y), cv::Point(x, FRAME_HEIGHT), cv::Scalar(0, 255, 0), 2);

	if (x - 25 > 0)
		cv::line(frame, cv::Point(x, y), cv::Point(x - 25, y), cv::Scalar(0, 255, 0), 2);
	else cv::line(frame, cv::Point(x, y), cv::Point(0, y), cv::Scalar(0, 255, 0), 2);

	if (x + 25 < FRAME_WIDTH)
		cv::line(frame, cv::Point(x, y), cv::Point(x + 25, y), cv::Scalar(0, 255, 0), 2);
	else cv::line(frame, cv::Point(x, y), cv::Point(FRAME_WIDTH, y), cv::Scalar(0, 255, 0), 2);

	cv::putText(frame, intToString(x) + "," + intToString(y), cv::Point(x, y + 30), 1, 1,
		cv::Scalar(0, 255, 0), 2); // coordinates show on screen.

}//drawObject


 // ================
 //       MAIN
 // ================
int main() {

	// 1. Read input images
	std::cout << "reading image 1" << endl;
	cv::Mat image1 = cv::imread("C:\\Users\\Arg0s2\\Pictures\\Camera Roll\\template.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image2 = cv::imread("C:\\Users\\Arg0s2\\Pictures\\Camera Roll\\model01.png", CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << "defining keypoints!" << endl;
	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cout << "constructing FAST" << endl;
	// 3. Define feature detector
	cv::Ptr<cv::FeatureDetector> ptrDetector;           // generic detector
	ptrDetector = cv::FastFeatureDetector::create(80);   // we select the FAST detector
	cout << "detecting keypoints" << endl;
	// 4. Keypoint detection
	ptrDetector->detect(image1, keypoints1);
	ptrDetector->detect(image2, keypoints2);

	std::cout << "Number of keypoints (image 1): " << keypoints1.size() << std::endl;
	std::cout << "Number of keypoints (image 2): " << keypoints2.size() << std::endl;

	// 5. Define a square neighborhood
	const int nsize(11); // size of the neighborhood
	cv::Rect neighborhood(0, 0, nsize, nsize); // 11x11
	cv::Mat patch1;
	cv::Mat patch2;

	// 6. For all keypoints in first image
	//    find best match in second image
	cv::Mat result;
	std::vector<cv::DMatch> matches;

	//for all keypoints in image 1
	for (int i = 0; i<keypoints1.size(); i++) {

		// define image patch
		neighborhood.x = (int)(keypoints1[i].pt.x - nsize / 2); // x component
		neighborhood.y = (int)(keypoints1[i].pt.y - nsize / 2); // y component


																// if neighborhood of points outside image, then continue with next point
		if (neighborhood.x<0 || neighborhood.y<0 ||
			neighborhood.x + nsize >= image1.cols || neighborhood.y + nsize >= image1.rows)
			continue;

		//patch in image 1
		patch1 = image1(neighborhood);

		// reset best correlation value;
		cv::DMatch bestMatch;

		//for all keypoints in image 2
		for (int j = 0; j<keypoints2.size(); j++) {

			// define image patch
			neighborhood.x = (int)(keypoints1[j].pt.x - nsize / 2); // x component
			neighborhood.y = (int)(keypoints1[j].pt.y - nsize / 2); // y component


																	// if neighborhood of points outside image, then continue with next point
			if (neighborhood.x<0 || neighborhood.y<0 ||
				neighborhood.x + nsize >= image2.cols || neighborhood.y + nsize >= image2.rows)
				continue;

			// patch in image 2
			patch2 = image2(neighborhood);

			// match the two patches
			cv::matchTemplate(patch1, patch2, result, CV_TM_CCORR_NORMED);

			// check if it is a best match
			if (result.at<float>(0, 0) < bestMatch.distance) {

				bestMatch.distance = result.at<float>(0, 0);
				bestMatch.queryIdx = i;
				bestMatch.trainIdx = j;
			}//if
		}//for

		 // add the best match
		matches.push_back(bestMatch);
	}//

	std::cout << "Number of matches: " << matches.size() << std::endl;

	// extract the 50 best matches
	std::nth_element(matches.begin(), matches.begin() + 10, matches.end());
	matches.erase(matches.begin() + 10, matches.end());

	std::cout << "Number of matches (after): " << matches.size() << std::endl;

	// Draw the matching results
	cv::Mat matchImage;
	cv::drawMatches(image1, keypoints1, // first image
		image2, keypoints2, // second image
		matches,     // vector of matches
		matchImage,  // produced image
		cv::Scalar(0, 255, 0),  // line color
		cv::Scalar(255, 0, 0)); // point color

								// Display the image of matches
	cv::namedWindow("Matches");
	cv::imshow("Matches", matchImage);

	// Match template

	// define a template
	cv::Mat target(image1);
	// Display the template
	//cv::namedWindow("Template");
	//cv::imshow("Template", target);

	// define search region
	cv::Mat roi(image2,
		// here top half of the image
		cv::Rect(0, 0, image2.cols, image2.rows));

	// perform template matching
	cv::matchTemplate(
		roi,    // search region
		target, // template
		result, // result
		CV_TM_CCORR); // similarity measure

					  // find most similar location
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

	// draw rectangle at most similar location
	// at minPt in this case
	cv::rectangle(roi, cv::Rect(minPt.x, minPt.y, target.cols, target.rows), 255);

	// Display the template
	cv::namedWindow("Best");
	cv::imshow("Best", image2);

	cv::waitKey();
	return 0;
}


