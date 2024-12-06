#include "Recorder.h"

#include "Defs.h"

#include <iostream>

int Recorder::start(cv::Size size) {
    if (!m_writer.open(m_output, kMJPG, kFPS, size)) {
        std::cerr << "error: failed to open VideoWriter.\n";
        return -1;
    }

    m_recording = true;
    m_thread = std::thread(&Recorder::do_write, this);

    return 0;
}

void Recorder::append(cv::Mat frame) {
    std::unique_lock lock(m_lock);
    m_queue.push(std::move(frame));
    m_cv.notify_all();
}

void Recorder::done() {
    m_recording = false;
    m_cv.notify_all();

    if (m_thread.joinable())
        m_thread.join();

    m_writer.release();
}

void Recorder::do_write() {
    while (true) {
        {
            std::unique_lock lock(m_lock);
            m_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !m_recording || !m_queue.empty();
            });
        }

        while (!m_queue.empty()) {
            cv::Mat frame;

            {
                std::unique_lock lock(m_lock);
                frame = std::move(m_queue.front());
                m_queue.pop();
            }

            m_writer << frame;
        }

        if (!m_recording && m_queue.empty())
            break;
    }
}
