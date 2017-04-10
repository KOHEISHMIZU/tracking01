#include<opencv2\opencv.hpp>
#include<opencv2\tracking\tracking.hpp>
#include <vector>
#include "stats.h"

#pragma comment(lib, "opencv_core320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")
#pragma comment(lib, "opencv_tracking320d.lib")

using namespace std;
using namespace cv;

void drawBoundingBox(Mat image, vector<Point2f> bb) {
	for (unsigned i = 0; i < bb.size() - 1; i++) {
		line(image, bb[i], bb[i + 1], Scalar(0, 0, 255), 2);
	}
	line(image, bb[bb.size() - 1], bb[0], Scalar(0, 0, 255), 2);
}

void targetline(Mat image, Mat homography, Mat out) {
	
	//対象物体画像のコーナー（角）の値を取得する
	if (!homography.empty()) {
		//homographyが空じゃないなら下の処理を行う
		vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(.0f, .0f);
		obj_corners[1] = Point2f(static_cast<float>(image.cols), .0f);
		obj_corners[2] = Point2f(static_cast<float>(image.cols), static_cast<float>(image.rows));
		obj_corners[3] = Point2f(.0f, static_cast<float>(image.rows));
		
		//射影を推定
		vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners,scene_corners, homography);

		//コーナー同士を線で結ぶ（シーン中のマップされた対象物体ーシーン画像）
		float w = static_cast<float>(image.cols);
		line(out, scene_corners[0] + Point2f(w, .0f), scene_corners[1] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[1] + Point2f(w, .0f), scene_corners[2] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[2] + Point2f(w, .0f), scene_corners[3] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[3] + Point2f(w, .0f), scene_corners[0] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
	}
}