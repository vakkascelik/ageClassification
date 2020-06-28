// opencvSample1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "opencv2/objdetect.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function for Face Detection 
void detectAndDraw(Net net, Mat frameOpenCVDNN);
string cascadeName, nestedCascadeName;

int main(int argc, char** argv)
{
	// VideoCapture class for playing video for which faces to be detected 
	VideoCapture capture;
	Mat frame, image;

	const std::string caffeConfigFile = "C:/Users/VakkasC/opencv/sources/samples/dnn/face_detector/deploy.prototxt";
	const std::string caffeWeightFile = "C:/Users/VakkasC/opencv/sources/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel";

	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

	// Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video 
	capture.open(0);
	if (capture.isOpened()) {
		// Capture frames from video and detect faces 
		cout << "Face Detection Started...." << endl;
		while (1) {
			capture >> frame;
			if (frame.empty())
				break;
			Mat frame1 = frame.clone();
			detectAndDraw(net,frame1);
			char c = (char)waitKey(10);

			// Press q to exit from window 
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}
	else
		cout << "Could not Open Camera";
	return 0;
}

void detectAndDraw(Net net, Mat frameOpenCVDNN) {
	
	Mat img2;
	resize(frameOpenCVDNN, img2, Size(300, 300));
	Mat inputBlob = blobFromImage(img2, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false);
	
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	int frameWidth = frameOpenCVDNN.cols;
	int frameHeight = frameOpenCVDNN.rows;

	float confidenceThreshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++)
	{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
				cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
			}
	}


	// Show Processed Image with detected faces 
	imshow("Face Detection", frameOpenCVDNN);
}
