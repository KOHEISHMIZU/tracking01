#ifndef FUNCTION_H
#define FUNCTION_H

#include<opencv2\opencv.hpp>
#include <vector>
#include "stats.h"

#pragma comment(lib, "opencv_core320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")

using namespace std;
using namespace cv;

void drawBoundingBox(Mat image, vector<Point2f> bb);
void targetline(Mat image, Mat homography, Mat out);
#endif
