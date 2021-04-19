#pragma once

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

/** Blob class (Also called object).
Blob, or objects, detected in the image. A blob can represent a car or a person for example.
*/
class Blob {
	public:
	int ID;
	int	firstFrameNumber;
	int lastFrameNumber;
	int frameCount;
	int avgWidth;
	int avgHeight;
	int maxWidth;
	int maxHeight;
	int collision;
	cv::Point currentPosition;
	cv::MatND currentHist;
	cv::Rect firstRectangle;
	cv::Rect lastRectangle;
	std::vector<int> contactContours;
	std::vector<cv::Mat> frames;
	std::vector<cv::MatND> histograms;
	
};