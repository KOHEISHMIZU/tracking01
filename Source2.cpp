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
	//画像の読み込み
	cv::Mat img1 = cv::imread("ウィンドラス.JPG", 1);
	cv::Mat img2 = cv::imread("shioji.jpg", 1);

	//キーポイント検出と特徴点記述
	cv::Ptr<cv::Feature2D> feature = cv::ORB::create();
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat desc1, desc2;

	feature->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
	feature->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

	if (desc2.rows == 0) {
		std::cout << "特徴点が検出できません" << std::endl;
		return -1;
	}

	//特徴点マッピング
	cv::BFMatcher matcher(cv::NORM_HAMMING);

	std::vector<std::vector<cv::DMatch>> knn_matches;

	matcher.knnMatch(desc1, desc2, knn_matches, 2);

	//対応点を絞る
	const double match_par = .6f;
	std::vector<cv::DMatch> good_matches;

	std::vector<cv::Point2f> match_point1;
	std::vector<cv::Point2f> match_point2;

	for (size_t i = 0; i < knn_matches.size(); ++i) {
		double dist1 = knn_matches[i][0].distance;
		double dist2 = knn_matches[i][1].distance;

		//良い点を残す
		if (dist1 <= dist2*match_par) {
			good_matches.push_back(knn_matches[i][0]);
			match_point1.push_back(kpts1[knn_matches[i][0].queryIdx].pt);
			match_point2.push_back(kpts2[knn_matches[i][0].trainIdx].pt);
		}
	}

	//ホモグラフィティ行列の作製
	cv::Mat masks;
	cv::Mat H;
	if (match_point1.size() != 0 && match_point2.size() != 0) {
		H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3.f);
	}

	//RANSACで使われた対応点のみ抽出
	std::vector<cv::DMatch> inlinerMatches;
	for (auto i = 0; i < masks.rows; ++i) {
		uchar*inliner = masks.ptr<uchar>(i);
		if (inliner[0] == 1) {
			inlinerMatches.push_back(good_matches[i]);
		}
	}
	//特徴点の表示
	cv::Mat out1;
	cv::Mat out2;
	cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, out1);
	cv::drawMatches(img1, kpts1, img2, kpts2, inlinerMatches, out2);

	//対象物体画像のコーナー（角）の値を取得する
	if (!H.empty()) {
		std::vector<cv::Point2f> obj_corners(4);
		obj_corners[0] = cv::Point2f(.0f, .0f);
		obj_corners[1] = cv::Point2f(static_cast<float>(img1.cols), .0f);
		obj_corners[2] = cv::Point2f(static_cast<float>(img1.cols), static_cast<float>(img1.rows));
		obj_corners[3] = cv::Point2f(.0f, static_cast<float>(img1.rows));

		//射影を推定
		std::vector<cv::Point2f> scene_corners(4);
		cv::perspectiveTransform(obj_corners, scene_corners, H);

		//コーナー同士を線で結ぶ（シーン中のマップされた対象物体ーシーン画像）
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

	//結果画像の表示
	imwrite("match_point.jpg", out1);
	imwrite("inlinermatch.jpg", out2);
	imshow("結果①", out1);
	imshow("結果②", out2);

	cv::waitKey(0);

	return 0;

}