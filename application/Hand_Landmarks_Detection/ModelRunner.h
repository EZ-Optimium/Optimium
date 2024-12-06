#pragma once

#include "InferEngine.h"
#include "Defs.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <thread>
#include <vector>

class ModelRunner final {
public:
    ~ModelRunner() noexcept { stop(); }

    void set_engine(InferEngine& engine) { m_engine = &engine; }

    bool is_running() const { return m_running; }

    int start();

    void stop();

    void infer();

    void update_data(const cv::Mat& frame);

    const std::vector<cv::Point>& landmarks() const {
        return m_landmarks[m_switch ? 0 : 1];
    }

    float average() const { return m_average; }

// private:
    void do_infer();

    InferEngine* m_engine = nullptr;

    // palm detection
    cv::Mat det_rgb_converted;
    cv::Mat det_padded;
    cv::Mat det_resized;
    cv::Mat det_input_data;

    // face landmark
    cv::Mat m_input_data;
    std::vector<cv::Point> m_landmarks[2];
    std::vector<cv::Rect> m_faces;
    bool m_switch = false;

    std::thread m_runner;
    std::atomic_flag m_wait = ATOMIC_FLAG_INIT;
    std::atomic<bool> m_run = true;
    std::atomic<bool> m_running = false;

    int64_t m_latencies[10] { 0, };
    int64_t m_counter = 0;
    float m_average = 0.0f;

    bool detected = true;
};