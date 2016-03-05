#pragma once
//OpenCV2 imports
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


//Printable imports
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

//Temp sleep
#include <conio.h>
#include <windows.h>


//Preprocess namespacing
using namespace cv;
using namespace std;

class UFinder {
	Mat src;
	float k_thresh = 160; //Contrast to thresh the white and black 
	float k_minSize = 15; //Minimum size for a shape to be
	float k_maxSize = 145; //Maximum size to process the image
	float k_minArea = 110; //Minimum area the object could have
	float k_maxArea = 900; //Maximum area the object could have
	float k_minLength = 120;//Minimum length for object to be
	float k_maxLength = 300; //Max length to contour to be
	float k_minSides = 5; //Minimum amount of sides the polygon needs
	float k_maxSides = 8; //Maximum amount of sides the polygon can have
	float k_minDepth = 41; //Minimum defect (Since it's a U we want the height as a defect)
	float k_maxDepth = 70; //Maximum for a indent in the shape to be
	float k_triggerDepth = 40; //Amount until to trigger it's a defect (Should be less than minDepth, to do anythin)
	float k_min_defects = 1; //Minimum of defects according to trigger
	float k_max_defects = 2; //Maximum of defects before it quits
	float k_obscure_depth = 1000; //If something went wrong in the calculation then calm down
public:
	void setThresh(float threshold_values) { k_thresh = threshold_values; }
	void setSize(float min, float max) { k_minSize = min; k_maxSize = max; }
	void setArea(float min, float max) { k_minArea = min; k_maxArea = max; }
	void setPerimeter(float min, float max) { k_minLength = min; k_maxLength = max; }
	void setSides(float min, float max) { k_minSides = min; k_maxSides = max; }
	void setDepth(float triggerDepth, float minDepth, float maxDepth, float minDefects, float maxDefects, float obscure_depth) 
	{
		k_triggerDepth = triggerDepth; k_minDepth = minDepth; k_maxSize = maxDepth; k_min_defects = minDefects; 
		k_max_defects = maxDefects; k_obscure_depth = obscure_depth;
	}
	void setImageByPath(char []);
	void setImageByBitmap(InputArray);
	Mat startProcess(float *, float *, float *, float *, const char *);
};

void UFinder::setImageByBitmap(InputArray buffer) {
	src = imdecode(buffer, 1);
}

void UFinder::setImageByPath(char path[]) {
	src = imread(path);
}

Mat UFinder::startProcess(float *x, float *y, float *fromCenter, float *pixelsCenter, const char *path = "_")
{
	Mat src_gray;
	RNG rng(12345);

	//Turn the image into grayscale and blur
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	src.release();
	//Show source window

	//Initialize threshold and contour variables 
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//First major step... Process threshold and put through contours
	threshold(src_gray, threshold_output, k_thresh, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	src_gray.release();

	//Draw a rectangle for debugging
	vector<RotatedRect> minRect(contours.size());

	//Image moments for details
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());

	//Check the hule (Connection)
	vector<vector<int>> hull(contours.size());
	vector<vector<Vec4i>> convDef(contours.size());

	//Add a drawing 
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

	//Loop through each contour
	for (size_t i = 0; i< contours.size(); i++)
	{
		//Second step, make sure the size is roughly to what we want
		if (contours[i].size() > k_minSize && contours[i].size() < k_maxSize)
		{
			printf("Pass size... %d\n", i);
			float area = contourArea(contours[i]);

			if (area > k_minArea && area < k_maxArea) {

				printf("Pass area... %d\n", i);
				float length = arcLength(contours[i], true); //The perimiter of the shape


				if (length > k_minLength && length < k_maxLength) {
					printf("Pass length(Perimiter)... %d\n", i);

					//Calculate polygon
					vector<Point> approx;
					approxPolyDP(contours[i], approx, 5, true);

					float sides = approx.size(); //Get the amount of sides on the polygon

					if (sides > k_minSides && sides < k_maxSides) {
						printf("Pass sides... %d\n", i);
						//Check for contour defects
						convexHull(contours[i], hull[i], false);
						convexityDefects(contours[i], hull[i], convDef[i]);

						bool passed = true;
						int amountOfDefects = 0;
						float defectDepth = 0;
						float finDepth = 0;

						printf("%d:\n", i);

						//Iterate through each defect in the shape
						for (int defect = 0; defect < convDef.size(); defect++) {
							defectDepth = (float)convDef[i][defect][3] / 256.0f; //Get distance in pixels and turn to float (Diagnols need to take into effect)
							printf("   Depth_Debug %.2f", defectDepth);
							printf("   \n");
							if (defectDepth > k_triggerDepth && defectDepth < k_obscure_depth) {
								if (defectDepth < k_minDepth || defectDepth > k_maxDepth) {
									passed = false;
									printf("\n\nError: Defect either too small or too large\n\n");
									break;
								}
								finDepth = defectDepth;
								amountOfDefects += 1;
							}
						}

						if (amountOfDefects < k_min_defects || amountOfDefects > k_max_defects) passed = false;


						//Only run if there is a U shape
						if (passed) {
							printf("Pass defects... %d\n", i);

							//Calculate area and point of contour
							mu[i] = moments(contours[i], false);

							//Mass center
							*x = (mu[i].m10 / mu[i].m00);
							*y = (mu[i].m01 / mu[i].m00);
							//Pixels from center
							*fromCenter = *x - *pixelsCenter;

							printf("Contour %d: Size - %d : Area - %.2f : Length - %.2f : Sides - %.2f : U depth - %.2f\n", i, contours[i].size(), area, length, sides, finDepth);

							if (path != "_") {
								mc[i] = Point2f(*x, *y);

								//Rectangle to draw
								minRect[i] = minAreaRect(Mat(contours[i]));

								Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); //Random color
																													  // contour drawing
								drawContours(drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());

								//Center of contour drawing
								circle(drawing, mc[i], 5, color, -1, 8, 0);

								// rotated rectangle
								Point2f rect_points[4]; minRect[i].points(rect_points);
								for (int j = 0; j < 4; j++) line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
							}
						}
					}
				}
			}
		}
	}
	if (path != "_") {
		imwrite(path, drawing);
	}
	printf("Done...\n");
	return drawing;
}

float getDistance(float centerX, float centerY) {
	//Center X is the correct pixel
	centerY -= 20; //Either subtract or add to get the correct pixel, you want it right under the U or this could be 
				   //Static so that you always get the depth of the bottom of the tower	
	return 0;
}

int main() {
	while (1) {
		try {
			printf("Started...\n");

			//Get the mass center of the U - Location wise
			float centerX = 0;
			float centerY = 0;
			//Pixels that the mass center is located from
			float fromCenter = 0;
			float pixelsWidth = 256; //X resolution divided by 2

			float distance = 0; //

			//Trigger 0 only when within these pixels
			float LeftTrigger = -15;
			float RightTrigger = 15;

			int direction[2] = { 0, 0 };

			UFinder find = UFinder();
			find.setImageByPath("C:/Users/David/Desktop/pics/farthestMiddle.bmp");
			find.setThresh(160);
			find.setArea(110, 1000);
			find.setPerimeter(120, 400);
			find.setSize(15, 145);
			find.setSides(3, 10);
			find.setDepth(40, 20, 90, 0, 10, 1000);
			find.startProcess(&centerX, &centerY, &fromCenter, &pixelsWidth, "C:\\Users\\David\\Desktop\\ok.jpeg");

			direction[0] = (fromCenter > LeftTrigger && fromCenter < RightTrigger) ? 0 : (fromCenter < LeftTrigger) ? 1 : 2; //Left right
			direction[1] = getDistance(centerX, centerY); //Put the distance calculation here (Forward backward)

			printf("CenterX: %.2f : CenterY: %.2f : fromCenter: %.2f : Direction: %d : Distance %.6f\n", centerX, centerY, fromCenter, direction[0], direction[1]);

		} catch (Exception ignored) {
			printf("Quiting cause of some error");
		}
		Sleep(1000);
	}

	return 0;
}
