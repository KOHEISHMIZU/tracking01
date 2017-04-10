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
	
	//�Ώە��̉摜�̃R�[�i�[�i�p�j�̒l���擾����
	if (!homography.empty()) {
		//homography���󂶂�Ȃ��Ȃ牺�̏������s��
		vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(.0f, .0f);
		obj_corners[1] = Point2f(static_cast<float>(image.cols), .0f);
		obj_corners[2] = Point2f(static_cast<float>(image.cols), static_cast<float>(image.rows));
		obj_corners[3] = Point2f(.0f, static_cast<float>(image.rows));
		
		//�ˉe�𐄒�
		vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners,scene_corners, homography);

		//�R�[�i�[���m����Ō��ԁi�V�[�����̃}�b�v���ꂽ�Ώە��́[�V�[���摜�j
		float w = static_cast<float>(image.cols);
		line(out, scene_corners[0] + Point2f(w, .0f), scene_corners[1] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[1] + Point2f(w, .0f), scene_corners[2] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[2] + Point2f(w, .0f), scene_corners[3] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
		line(out, scene_corners[3] + Point2f(w, .0f), scene_corners[0] + Point2f(w, .0f), cv::Scalar(0, 0, 255), 2);
	}
}