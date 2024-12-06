#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "nms.h"

void decode_box(float* raw_data, float* anchors);

float* loadAnchors(const std::string& filePath);

void sigmoid_score(float* raw_data);

void decodeBoundingBoxes(
    float* decodedBoxes, 
    const float* rawBoxes, 
    const float* anchors
);

std::vector<cv::Point2f> getTriangle(
    const cv::Point2f& kp0, 
    const cv::Point2f& kp2, 
    float side, 
    float boxShift
);

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractHandDetails(
    const BoundBox& detect,
    cv::Point2f offset
);

cv::Mat computeAffineMatrix(
    const std::vector<cv::Point2f>& sourceTriangle, 
    float scale
);

void warpHandRegion(
    const cv::Mat& imgPad, 
    const cv::Mat& affineMatrix,
    cv::Mat &output
);

std::vector<std::array<float, 3>> extractLandmarks(const float* outraw);

cv::Mat padAffineMatrix(const cv::Mat& affineMatrix);

cv::Mat computeInverseMatrix(const cv::Mat& paddedMatrix);

std::vector<cv::Point> projectLandmarksToOriginal(
    const std::vector<std::array<float, 3>>& keypoints,
    const cv::Mat& inverseMatrix,
    const cv::Size& padding
);