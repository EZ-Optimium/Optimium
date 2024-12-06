#include "ModelRunner.h"
#include "Defs.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <numeric>
#include <iterator>
#include <iostream>

using timer = std::chrono::high_resolution_clock;

int ModelRunner::start() {
    if (m_engine == nullptr) {
        std::cerr << "error: engine not set.\n";
        return 1;
    }

    m_landmarks[0].resize(kNumLandmarks);
    m_landmarks[1].resize(kNumLandmarks);

    m_wait.test_and_set();
    m_runner = std::thread(&ModelRunner::do_infer, this);

    return 0;
}

void ModelRunner::stop() {
    m_run = false;
    m_wait.clear();

    if (m_runner.joinable())
        m_runner.join();
}

void ModelRunner::infer() {
    m_wait.clear();
}

void ModelRunner::update_data(const cv::Mat& frame) {
    cv::cvtColor(frame, det_rgb_converted, cv::COLOR_BGR2RGB);
    cv::copyMakeBorder(det_rgb_converted, m_input_data, detPadHeight, detPadHeight, detPadWidth, detPadWidth, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::resize(m_input_data, det_resized, cv::Size(detInputSize, detInputSize));
    det_resized.convertTo(det_input_data, CV_32F, 1.0 / 255.0);
}

void ModelRunner::do_infer() {
    while (m_run) {
        while (m_wait.test_and_set())
            ; // busy wait

        if (!m_run) break;

        m_running = true;

        auto& next_landmark = m_landmarks[m_switch ? 1 : 0];
        next_landmark.clear();
        m_faces.clear();

        auto begin = timer::now();

        detected = m_engine->do_infer(det_input_data, m_input_data, next_landmark, m_faces);
        auto end = timer::now();

        m_latencies[m_counter++ % 10]  = (end - begin).count();
        if (m_counter > 10) {
            m_average = std::accumulate(std::begin(m_latencies), std::end(m_latencies), int64_t(0)) / (10 * 1000000.0f);
        }

        m_switch = !m_switch;
        m_running = false;

    }
}
