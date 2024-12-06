#include "InferEngine.h"
#include "Defs.h"
#include "Recorder.h"
#include "ModelRunner.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cstdarg>
#include <ctime>
#include <chrono>
#include <thread>
#include <iostream>

#include <dirent.h>
#include <sys/stat.h>

using timer = std::chrono::high_resolution_clock;

using ms = std::chrono::milliseconds;

inline ms to_ms(timer::duration d) {
    return std::chrono::duration_cast<ms>(d);
}

enum class Kind {
    TFLite,
    Optimium
};

const auto help_message = R"(
Optimium Demo App Command:
  - 'l' : Live demo mode
  - 'd' : Diffrentiate mode
  - 'r' : Show previous record
  - 'q' : Quit the app

Select the command:
)";

const static std::array<cv::Point, 21> Vertexes = {
    // Thumb
    cv::Point{0, 1},
    cv::Point{1, 2},
    cv::Point{2, 3},
    cv::Point{3, 4},

    // Index finger
    cv::Point{5, 6},
    cv::Point{6, 7},
    cv::Point{7, 8},

    // Middle finger
    cv::Point{9, 10},
    cv::Point{10, 11},
    cv::Point{11, 12},

    // Ring finger
    cv::Point{13, 14},
    cv::Point{14, 15},
    cv::Point{15, 16},

    // Pinky finger
    cv::Point{17, 18},
    cv::Point{18, 19},
    cv::Point{19, 20},

    // Palm connections
    cv::Point{0, 5},
    cv::Point{5, 9},
    cv::Point{9, 13},
    cv::Point{13, 17},
    cv::Point{0, 17}
};

const static auto kTextColor = CV_RGB(255, 0, 190);
const static auto kVertexColor = CV_RGB(0, 100, 255);
const static auto kEdgeColor = CV_RGB(0, 255, 30);

std::unique_ptr<InferEngine> tflite;
std::unique_ptr<InferEngine> optimium;

int initialize();
void finalize();
int run_live_demo();
int run_diff_demo();

// save camera configurations
double zoom = 130;
std::string prev_record{};

static int show(const std::string& file_name);

std::string format(const char* format, ...) {
    va_list args;
    va_start(args, format);

    char buffer[256];
    auto len = vsnprintf(buffer, sizeof(buffer), format, args);

    va_end(args);

    return std::string(buffer, len);
}

std::string find_latest_record() {
    auto* dir = opendir("outputs");

    if (dir == nullptr) {
        std::cerr << "failed to open directory: " << strerror(errno) << "\n";
        return {};
    }

    std::string name;

    do {
        errno = 0;
        auto* entry = readdir(dir);

        if (entry == nullptr) {

            if (errno != 0) {
                std::cerr << "failed to read directory: " << strerror(errno) << "\n";
                closedir(dir);
                return {};
            }

            // EOF
            break;
        }
        
        if (strncmp("record_result_", entry->d_name, 14) != 0)
            continue;

        if (name.empty())
            name = entry->d_name;

        if (name.compare(entry->d_name) < 0)
            name = entry->d_name; 
    } while (true);
    
    closedir(dir);

    if (name.empty())
        return {};

    return "outputs/" + name;
}

int main() {
    if (initialize()) {
        finalize();
        return -1;
    }

    bool run = true;
    while (run) {
        int cmd;

        std::cout << help_message;
        
        while (true) {
            cmd = std::cin.get();
            if (cmd == '\r' || cmd == '\n' || cmd == '\t' || cmd == ' ')
                continue;

            break;
        }

        switch (cmd) {
            default:
                std::cout << "unknown command '" << (char)cmd << "'\n";
                break;

            case 'l':
                if (run_live_demo())
                    run = false;
                break;

            case 'd':
                if (run_diff_demo())
                    run = false;
                break;

            case 'r': {
                auto video = find_latest_record();

                if (video.empty()) {
                    std::cout << "record is empty.\n";
                } else {
                    std::cout << "latest record: " << video << "\n";
                    show(video);
                }
                break;
            }

            case 'q':
                run = false;
                break;
        }
    }

    finalize();
    
    return 0;
}

int initialize() {
    // create directory
    if (mkdir("outputs", 0755) < 0 && errno != EEXIST) {
        std::cerr << "error: failed to create directory: " << strerror(errno) << "\n";
        return 1;
    }

    // Loading tflite
    tflite = InferEngine::create_tflite_engine();
    if (!tflite)
        return 1;

    optimium = InferEngine::create_optimium_engine();
    if (!optimium)
        return 1;

    return 0;
}

void finalize() {
    tflite.reset();
    optimium.reset();
}

static void config_reader(cv::VideoCapture& capture) {
    if (!capture.isOpened())
        return;
    capture.set(cv::CAP_PROP_FOURCC, kYUYV);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, kWidth);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
}

static void render_landmarks(cv::Mat& frame, const std::vector<cv::Point>& landmarks) {
    for (const auto [start, end] : Vertexes) {
        const auto& start_landmark = landmarks[start];
        const auto& end_landmark = landmarks[end];

        cv::line(frame, start_landmark, end_landmark, kEdgeColor, 3);
    }

    for (const auto& landmark : landmarks)
        cv::circle(frame, landmark, 5, kVertexColor, -1);
}

static void render_text(cv::Mat& frame, Kind kind, float latency) {
    char text_buffer[128];
    const char* text = (kind == Kind::TFLite) ? "current: TFLite" : "current: Optimium";

    if (latency == 0) {
        snprintf(text_buffer, sizeof(text_buffer), "model latency: 0.0ms / FPS -");
    } else {
        snprintf(text_buffer, sizeof(text_buffer), "model latency: %.02fms / FPS: %.02f", latency, 1000 / latency);
    }

    cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, kTextColor);
    cv::putText(frame, text_buffer, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.0, kTextColor);
}


int run_live_demo() {
    cv::VideoCapture reader(0);
    if (!reader.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    config_reader(reader);

    ModelRunner runner;
    Kind kind = Kind::TFLite;

    cv::Mat current, prev;

    // set default engine: tflite
    runner.set_engine(*tflite);
    runner.start();

    auto time_point = timer::now();
    bool run = true;

    while (run) {
        if (!reader.read(current)) {
            std::cerr << "error: camera read error.\n";
            return 1;
        }
        
        // transform data
        runner.update_data(current);

        // start inference if model is not running.
        if (!runner.is_running())
            runner.infer();

        auto now = timer::now();
        auto delay = 33 - to_ms(now - time_point).count();

        if (delay > 0)
            std::this_thread::sleep_for(ms(delay));

        if (prev.empty()) {
            std::swap(current, prev);
            continue;
        }

        if (runner.detected)
            render_landmarks(prev, runner.landmarks());
        render_text(prev, kind, runner.average());

        cv::imshow("Demo", prev);

        std::swap(current, prev);

        auto key = cv::waitKey(1);
        switch (key) {
            default:
                // do nothing
                break;

            case 's': {
                // switch model
                kind = (kind == Kind::TFLite) ? Kind::Optimium : Kind::TFLite;

                if (kind == Kind::TFLite) {
                    runner.set_engine(*tflite);
                } else {
                    runner.set_engine(*optimium);
                }
                break;
            }

            case '+': case '=': {
                zoom += 10;
                reader.set(cv::CAP_PROP_ZOOM, zoom);
                zoom = reader.get(cv::CAP_PROP_ZOOM);
                std::cerr << "zoom: " << zoom << "\n";
                break;
            }

            case '-': {
                zoom -= 10;
                reader.set(cv::CAP_PROP_ZOOM, zoom);
                zoom = reader.get(cv::CAP_PROP_ZOOM);
                std::cerr << "zoom: " << zoom << "\n";
                break;
            }

            case 'q':
                run = false;
                break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}


static int record(const std::string& timestamp) {
    cv::VideoCapture reader;
    auto file = format("outputs/record_data_%s.avi", timestamp.c_str());
    Recorder recorder(file);

    if (!reader.open("/dev/video0", cv::CAP_V4L2)) {
        std::cerr << "error: failed to open camera.\n";
        return 1;
    }

    config_reader(reader);

    cv::Mat frame, show_frame;

    auto time_point = timer::now();
    bool run = true;
    bool recording = false;

    while (run) {
        if (!reader.read(frame)) {
            std::cerr << "error: camera read error.\n";
            return 1;
        }

        frame.copyTo(show_frame);
        if (recording)
            recorder.append(std::move(frame));

        auto now = timer::now();
        auto delay = 33 - to_ms(now - time_point).count();
        time_point = now;

        if (delay > 0)
            std::this_thread::sleep_for(ms(delay));

        cv::putText(show_frame, recording ? "Recording..." : "Idle...", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, recording ? CV_RGB(255, 0, 0) : CV_RGB(0, 255, 0));

        cv::imshow("Demo", show_frame);

        auto key = cv::waitKey(1);
        switch (key) {
            default:
                // do nothing
                break;

            case 'r':
                recording = !recording;
                if (recording) {
                    if (recorder.start()) {
                        std::cerr << "failed to start recorder.\n";
                        return 1;
                    }
                } else {
                    recorder.done();
                }
                break;

            case '+': case '=': {
                zoom += 10;
                reader.set(cv::CAP_PROP_ZOOM, zoom);
                zoom = reader.get(cv::CAP_PROP_ZOOM);
                std::cerr << "zoom: " << zoom << "\n";
                break;
            }

            case '-': {
                zoom -= 10;
                reader.set(cv::CAP_PROP_ZOOM, zoom);
                zoom = reader.get(cv::CAP_PROP_ZOOM);
                std::cerr << "zoom: " << zoom << "\n";
                break;
            }

            case 'q':
                run = false;
                break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}

static int record_model(InferEngine& engine, Kind kind, const std::string& timestamp) {
    cv::VideoCapture reader;
    auto input_name = format("outputs/record_data_%s.avi", timestamp.c_str());

    if (!reader.open(input_name, cv::CAP_FFMPEG)) {
        std::cerr << "error: failed to open " << input_name << ".\n";
        return 1;
    }

    auto output_name = format("outputs/record_%s_%s.avi", (kind == Kind::TFLite) ? "tflite" : "optimium", timestamp.c_str());

    Recorder recorder(output_name);
    recorder.start();

    config_reader(reader);

    ModelRunner runner;

    cv::Mat current, prev;

    runner.set_engine(engine);
    runner.start();

    auto time_point = timer::now();
    bool run = true;

    while (run) {
        if (!reader.read(current)) {
            // done
            break;
        }

        // transform data
        runner.update_data(current);

        // start inference if model is not running.
        if (!runner.is_running())
            runner.infer();

        auto now = timer::now();
        auto delay = 33 - to_ms(now - time_point).count();
        time_point = now;

        if (delay > 0)
            std::this_thread::sleep_for(ms(delay));

        if (prev.empty()) {
            std::swap(current, prev);
            continue;
        }

        if (runner.detected)
            render_landmarks(prev, runner.landmarks());
        render_text(prev, kind, runner.average());

        recorder.append(std::move(prev));

        std::swap(current, prev);
    }

    return 0;
}

static int concat_model(const std::string& timestamp) {
    auto tflite_file = format("outputs/record_tflite_%s.avi", timestamp.c_str());
    auto optimium_file = format("outputs/record_optimium_%s.avi", timestamp.c_str());
    auto result_file = format("outputs/record_result_%s.avi", timestamp.c_str());

    cv::VideoCapture tflite(tflite_file);
    cv::VideoCapture optimium(optimium_file);

    if (!tflite.isOpened()) {
        std::cerr << "failed to open " << tflite_file << ".\n";
        return 1;
    }

    if (!optimium.isOpened()) {
        std::cerr << "failed to open " << optimium_file << ".\n";
        return 1;
    }

    Recorder recorder(result_file);
    
    constexpr auto W = (kWidth / 3) * 2;
    constexpr auto H = (kHeight / 3) * 2;

    recorder.start(cv::Size(W * 2, H));

    while (true) {
        cv::Mat tflite_frame, optimium_frame;

        tflite >> tflite_frame;
        optimium >> optimium_frame;

        if (tflite_frame.empty() && optimium_frame.empty())
            break;

        cv::resize(tflite_frame, tflite_frame, cv::Size(W, H));
        cv::resize(optimium_frame, optimium_frame, cv::Size(W, H));

        cv::Mat output;
        cv::hconcat(tflite_frame, optimium_frame, output);
        recorder.append(std::move(output));
    }

    recorder.done();

    return 0;
}

static int show(const std::string& file_name) {
    cv::VideoCapture video(file_name);

    if (!video.isOpened()) {
        std::cerr << "failed to open " << file_name << ".\n";
        return 1;
    }

    cv::Mat frame;
    auto time_point = timer::now();
    int delta = 100;
    bool pause = false;
    bool run = true;

    while (run) {
        if (!pause)
            video >> frame;

        if (frame.empty()) {
            // rewind frame
            video.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        auto now = timer::now();
        auto delay = delta - to_ms(now - time_point).count();
        time_point = now;

        if (delay > 0)
            std::this_thread::sleep_for(ms(delay));

        cv::imshow("Demo", frame);

        auto key = cv::waitKey(1);
        switch (key) {
            default:
                // ignore
                break;

            case 'q':
                run = false;
                break;

            case ' ':
                pause = !pause;
                break;
            
            case ',':
                delta = std::min(1000, delta + 10);
                std::cerr << "delta: " << delta << "\n";
                break;

            case '.':
                delta = std::max(33, delta - 10);
                std::cerr << "delta: " << delta << "\n";
                break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}

int run_diff_demo() {
    std::string timestamp;
    {
        char buffer[128];
        struct tm now;

        auto t = time(nullptr);
        localtime_r(&t, &now);

        auto len = strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", &now);
        timestamp = std::string(buffer, len);
    }

    if (auto ret = record(timestamp); ret)
        return ret;

    std::cerr << "processing data on TFLite...\n";
    if (auto ret = record_model(*tflite, Kind::TFLite, timestamp); ret)
        return ret;

    std::cerr << "processing data on Optimium...\n";
    if (auto ret = record_model(*optimium, Kind::Optimium, timestamp); ret)
        return ret;

    std::cerr << "processing...\n";
    concat_model(timestamp);

    std::cerr << "done.\n";

    auto file_name = format("outputs/record_result_%s.avi", timestamp.c_str());
    show(file_name);

    return 0;
}

