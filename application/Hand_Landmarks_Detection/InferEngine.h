#pragma once

#include <opencv2/core.hpp>

#include <vector>
#include <memory>

class InferEngine {
public:
    virtual ~InferEngine() noexcept = default;

    virtual bool do_infer(const cv::Mat& det_input, cv::Mat& land_input, std::vector<cv::Point>& landmarks, std::vector<cv::Rect>& faces) = 0;

    static std::unique_ptr<InferEngine> create_tflite_engine();
    static std::unique_ptr<InferEngine> create_optimium_engine();
    float average() const { return models_average;}
    int64_t models_latencies[10] {0, };
    int64_t models_counter = 0;
    float models_average = 0.0f;

};