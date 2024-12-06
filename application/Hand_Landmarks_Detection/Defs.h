#pragma once

#include <opencv2/videoio.hpp>

constexpr size_t kNumLandmarks = 468;
constexpr int kWidth = 640;
constexpr int kHeight = 480;

// palm detection 
// 1920x1080 => 128 x 72 (1080*127/1920) - keeping ratio
constexpr int detPadHeight = 80; //  (640 - kHeight) / 2
constexpr int detPadWidth = 0; //  (640 - kWidth) / 2
constexpr int detInputSize = 192;
constexpr float detInputSizeF = static_cast<float>(detInputSize);
constexpr int detclnum = 2016;
constexpr float confidenceThreshold = 0.5;
constexpr float minSuppressionThreshold = 0.3;
constexpr float boxEnlarge = 1;
constexpr float boxShift = 0.2;

// hand landmark
constexpr int kInputSize = 224;
constexpr float kInputSizeF = static_cast<float>(kInputSize);

static const auto kMJPG = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
static const auto kYUYV = cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V');
constexpr auto kFPS = 30.0f;
constexpr auto kPerFrameMS = 16;

constexpr auto kTFLite = false;
constexpr auto kOptimium = true;