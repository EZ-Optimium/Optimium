#pragma once

#include "Defs.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <string>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <queue>

class Recorder final {
public:
    explicit Recorder(std::string output)
        : m_output(std::move(output)) {}

    ~Recorder() noexcept { done(); }

    int start(cv::Size size = cv::Size(kWidth, kHeight));
    void append(cv::Mat frame);
    void done();

private:
    std::string m_output;

    cv::VideoWriter m_writer;
    std::thread m_thread;
    std::condition_variable m_cv;
    std::mutex m_lock;
    std::queue<cv::Mat> m_queue;
    std::atomic<bool> m_recording;

    void do_write();
}; // end class Recorder
