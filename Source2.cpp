#include<opencv2\opencv.hpp>
#include<iostream>
#include<vector>

#pragma comment(lib, "opencv_features2d320d.lib")
#pragma comment(lib, "opencv_core320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_highgui320d.lib")
#pragma comment(lib, "opencv_calib3d320d.lib")
int main(int argc, char* argv[])
{
	//�摜�̓ǂݍ���
	cv::Mat img1 = cv::imread("�E�B���h���X.JPG", 1);
	cv::Mat img2 = cv::imread("shioji.jpg", 1);

	//�L�[�|�C���g���o�Ɠ����_�L�q
	cv::Ptr<cv::Feature2D> feature = cv::ORB::create();
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat desc1, desc2;

	feature->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
	feature->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

	if (desc2.rows == 0) {
		std::cout << "�����_�����o�ł��܂���" << std::endl;
		return -1;
	}

	//�����_�}�b�s���O
	cv::BFMatcher matcher(cv::NORM_HAMMING);

	std::vector<std::vector<cv::DMatch>> knn_matches;

	matcher.knnMatch(desc1, desc2, knn_matches, 2);

	//�Ή��_���i��
	const double match_par = .6f;
	std::vector<cv::DMatch> good_matches;

	std::vector<cv::Point2f> match_point1;
	std::vector<cv::Point2f> match_point2;

	for (size_t i = 0; i < knn_matches.size(); ++i) {
		double dist1 = knn_matches[i][0].distance;
		double dist2 = knn_matches[i][1].distance;

		//�ǂ��_���c��
		if (dist1 <= dist2*match_par) {
			good_matches.push_back(knn_matches[i][0]);
			match_point1.push_back(kpts1[knn_matches[i][0].queryIdx].pt);
			match_point2.push_back(kpts2[knn_matches[i][0].trainIdx].pt);
		}
	}

	//�z���O���t�B�e�B�s��̍쐻
	cv::Mat masks;
	cv::Mat H;
	if (match_point1.size() != 0 && match_point2.size() != 0) {
		H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3.f);
	}

	//RANSAC�Ŏg��ꂽ�Ή��_�̂ݒ��o
	std::vector<cv::DMatch> inlinerMatches;
	for (auto i = 0; i < masks.rows; ++i) {
		uchar*inliner = masks.ptr<uchar>(i);
		if (inliner[0] == 1) {
			inlinerMatches.push_back(good_matches[i]);
		}
	}
	//�����_�̕\��
	cv::Mat out1;
	cv::Mat out2;
	cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, out1);
	cv::drawMatches(img1, kpts1, img2, kpts2, inlinerMatches, out2);

	//�Ώە��̉摜�̃R�[�i�[�i�p�j�̒l���擾����
	if (!H.empty()) {
		std::vector<cv::Point2f> obj_corners(4);
		obj_corners[0] = cv::Point2f(.0f, .0f);
		obj_corners[1] = cv::Point2f(static_cast<float>(img1.cols), .0f);
		obj_corners[2] = cv::Point2f(static_cast<float>(img1.cols), static_cast<float>(img1.rows));
		obj_corners[3] = cv::Point2f(.0f, static_cast<float>(img1.rows));

		//�ˉe�𐄒�
		std::vector<cv::Point2f> scene_corners(4);
		cv::perspectiveTransform(obj_corners, scene_corners, H);

		//�R�[�i�[���m����Ō��ԁi�V�[�����̃}�b�v���ꂽ�Ώە��́[�V�[���摜�j
		float w = static_cast<float>(img1.cols);
		cv::line(out1, scene_corners[0] + cv::Point2f(w, .0f), scene_corners[1] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out1, scene_corners[1] + cv::Point2f(w, .0f), scene_corners[2] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out1, scene_corners[2] + cv::Point2f(w, .0f), scene_corners[3] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out1, scene_corners[3] + cv::Point2f(w, .0f), scene_corners[0] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);

		cv::line(out2, scene_corners[0] + cv::Point2f(w, .0f), scene_corners[1] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out2, scene_corners[1] + cv::Point2f(w, .0f), scene_corners[2] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out2, scene_corners[2] + cv::Point2f(w, .0f), scene_corners[3] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
		cv::line(out2, scene_corners[3] + cv::Point2f(w, .0f), scene_corners[0] + cv::Point2f(w, .0f), cv::Scalar(0, 255, 0), 2);
	}

	//���ʉ摜�̕\��
	imwrite("match_point.jpg", out1);
	imwrite("inlinermatch.jpg", out2);
	imshow("���ʇ@", out1);
	imshow("���ʇA", out2);

	cv::waitKey(0);

	return 0;

}