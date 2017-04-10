#include<opencv2\opencv.hpp>
#include<opencv2\tracking\tracking.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include "stats.h"
#include "function1.h"

#pragma comment(lib, "opencv_core320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")
#pragma comment(lib, "opencv_videoio320d.lib")
#pragma comment(lib, "opencv_tracking320d.lib")
#pragma comment(lib, "opencv_features2d320d.lib")
#pragma comment(lib, "opencv_highgui320d.lib")
#pragma comment(lib, "opencv_calib3d320d.lib")

using namespace std;
using namespace cv;


class tracker
	{
	public:
		tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
			detector(_detector),
			matcher(_matcher)
		{}

		void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
		Mat process(const Mat frame, Stats& stats);
		Ptr<Feature2D> getDetector() {
			return detector;
		}
	protected:
		Ptr<Feature2D> detector;
		Ptr<DescriptorMatcher> matcher;
		Mat first_frame, first_desc;
		vector<KeyPoint> first_kp;
		vector<Point2f> object_bb;

};

void tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
	{
		first_frame = frame.clone();
		detector->detectAndCompute(first_frame, noArray(), first_kp, first_desc);
		stats.keypoints = (int)first_kp.size();
		drawBoundingBox(first_frame, bb);
		putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
		object_bb = bb;
	}

	Mat tracker::process(const Mat frame, Stats& stats)
	{
		vector<KeyPoint> kp;
		Mat desc;
		detector->detectAndCompute(frame, noArray(), kp, desc);
		stats.keypoints = (int)kp.size();

		//対応点を絞る
		const double match_par = .6f;
		vector<DMatch> good_matches;
		vector< vector<DMatch> > matches;
		vector<Point2f> matched1, matched2;
		matcher->knnMatch(first_desc, desc, matches, 2);

		for (size_t i = 0; i < matches.size(); ++i) {
			double dist1 = matches[i][0].distance;
			double dist2 = matches[i][1].distance;

			//良い点を残す
			if (dist1 <= dist2*match_par) {
				good_matches.push_back(matches[i][0]);
				matched1.push_back(first_kp[matches[i][0].queryIdx].pt);
				matched2.push_back(kp[matches[i][0].trainIdx].pt);
			}
		}
		//ホモグラフィティ行列の作製
		cv::Mat masks;
		cv::Mat H;
		if (matched1.size() != 0 && matched2.size() != 0) {
			H = findHomography(matched1, matched2, masks, cv::RANSAC, 3.f);
		}
		//RANSACで使われた対応点のみ抽出
		vector<DMatch> inlinerMatches;
		for (auto i = 0; i < masks.rows; ++i) {
			uchar*inliner = masks.ptr<uchar>(i);
			if (inliner[0] == 1) {
				inlinerMatches.push_back(good_matches[i]);
			}
		}
		vector<Point2f> new_bb;
		perspectiveTransform(object_bb, new_bb, H);
		Mat frame_with_bb = frame.clone();
		Mat res;
		drawMatches(first_frame, first_kp, frame_with_bb, kp, inlinerMatches, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
		targetline(frame_with_bb, H, res);
		return res;
	}

	int main(int argc, char **argv)
	{
		VideoCapture video_in(argv[1]);
		if (!video_in.isOpened()) {
			cout << "ビデオが開けません" << endl;
			return -1;
		}
		std::string video_name = argv[1];

		Stats stats, orb_stats;
		Ptr<ORB> feature = ORB::create();
		feature->setMaxFeatures(stats.keypoints);
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		tracker orb_tracker(feature, matcher);

		Mat frame;
		video_in >> frame;

		//namedWindow(video_name, WINDOW_NORMAL);
		//resizeWindow(video_name, frame.cols, frame.rows);

		vector<Point2f> bb;
		cv::Rect2d uBox = selectROI(video_name, frame);
		bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
		bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
		bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
		bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

		orb_tracker.setFirstFrame(frame, bb, "ORB", stats);
		Mat orb_res;
		
		while (1) {
			video_in >> frame;
			if (frame.empty())
				break;
			orb_res = orb_tracker.process(frame, stats);
			imshow(video_name, orb_res);

			int key = cv::waitKey(30);
			if (key == 0x1b)//Push Esc Key
				break;
		}

		return 0;
}
