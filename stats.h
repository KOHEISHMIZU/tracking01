#ifndef STATS_H
#define STATS_H

struct Stats
{
	int matches;
	int inliners;
	double ratio;
	int keypoints;

	Stats():matches(0),inliners(0),ratio(0),keypoints(0){}
};
#endif