#include "InferEngine.h"
#include "Defs.h"
#include "Postprocess.h"
#include "nms.h"

#include <chrono>
#include <Optimium/Runtime.h>
#include <Optimium/Runtime/Logging/LogSettings.h>
#include <Optimium/Runtime/Utils/StreamHelper.h>

#include <iostream>
#include <numeric>
#include <iterator>

#include <opencv2/opencv.hpp>

constexpr auto OptimiumDetModelPath = "palm_detection_lite.model";
constexpr auto OptimiumLandmarkModelPath = "hand_landmark_lite.model";

constexpr auto AnchorsPath = "anchors.csv";

namespace rt = optimium::runtime;

using timer = std::chrono::high_resolution_clock;

template <typename T>
inline std::ostream &operator <<(std::ostream &OS, const std::vector<T>& vec) {
    for (auto i = 0; i < vec.size(); ++i) {
        OS << vec[i];

        if (i + 1 < vec.size())
            OS << ", ";
    }

    return OS;
}

inline std::ostream &operator <<(std::ostream &OS, const cv::Point2f &p) {
    return OS << "(x=" << p.x << ", y=" << p.y << ")";
}


class OptimiumInferEngine final : public InferEngine {
public:
    rt::Result<void> init() {
        rt::LogSettings::addWriter(rt::WriterOption::FileWriter("optimium_runtime.log"));
        rt::LogSettings::setLogLevel(rt::LogLevel::Debug);
        context = TRY(rt::Context::create());

        det_options.ThreadsCount = 2;
        det_model = TRY(context.loadModel(OptimiumDetModelPath, rt::ArrayRef<rt::Device>(), det_options));
        det_request = TRY(det_model.createRequest());

        m_options.ThreadsCount = 2;
        m_model = TRY(context.loadModel(OptimiumLandmarkModelPath, rt::ArrayRef<rt::Device>(), m_options));
        m_request = TRY(m_model.createRequest());

        auto output_info = TRY(m_model.getOutputTensorInfo("Identity"));
        m_output_size = output_info.TensorShape.getTotalElementCount();

        auto regressors_info = TRY(m_model.getOutputTensorInfo(0));
        auto classificators_info = TRY(m_model.getOutputTensorInfo(1));
        anchors = loadAnchors(AnchorsPath);

        return rt::Ok();
    }

    bool do_infer(const cv::Mat& data, cv::Mat& land_data, std::vector<cv::Point>& landmarks, std::vector<cv::Rect>& faces) override {
        auto result = [&]() -> rt::Result<void> {

            handDetected = false;
            // detection
            {
                auto det_input_tensor = TRY(det_request.getInputTensor("input_1"));
                auto det_input_buffer = det_input_tensor.getRawBuffer();
                auto* det_raw = det_input_buffer.cast<float>();

                cv::Mat det_input(detInputSize, detInputSize, CV_32FC3, det_raw);
                data.copyTo(det_input);
            }
            auto begin = timer::now();
            CHECK(det_request.infer());
            CHECK(det_request.wait());
            auto palm_end = timer::now();
            auto palm_time = (palm_end - begin).count();
            
            auto box_tensor = TRY(det_request.getOutputTensor(0));
            auto score_tensor = TRY(det_request.getOutputTensor(1));
            auto box_buffer = box_tensor.getRawBuffer();
            auto score_buffer = score_tensor.getRawBuffer();
            auto* rawBoxes = box_buffer.cast<float>();
            auto* rawScores = score_buffer.cast<float>();

            // Apply sigmoid to confidence scores
            sigmoid_score(rawScores);

            // Decode boxes with pre-defined anchors
            auto *decodedBoxes = rawBoxes;
            for (auto i = 0; i < detclnum; ++i) {
                const float* anchor = &anchors[i * 4];      // Each anchor has 4 values
                float* decodedBox = &decodedBoxes[i * 18];  // Each decoded box also has 18 values

                decodedBox[0] += anchor[0] * 192;           // dx + anchor_x * input size
                decodedBox[1] += anchor[1] * 192;           // dy + anchor_y * input size
            }

            // Filter out boxes with confidence threshold
            candidateDetect.clear();
            filteredProbabilities.clear();
            indices.clear();
            for (auto i = 0; i < detclnum; ++i) {
                if (rawScores[i] <= confidenceThreshold)
                    continue;
                candidateDetect.emplace_back(decodedBoxes + i * 18);
                filteredProbabilities.push_back(rawScores[i]);
                indices.push_back(i);
            }

            // Perform Non-Maximum Suppression (NMS) - Pick the first detected hand by default
            boxIds.clear();
            boxIds = nonMaximumSuppression(candidateDetect, filteredProbabilities);

            // If at least one box is detected, proceed palm(hand) detection
            if (!boxIds.empty()){
                int boxId = boxIds[0];
                BoundBox detect = candidateDetect[boxId];
                auto *ptr = anchors + 4 * indices[boxId];
                cv::Point2f center_wo_offset(*ptr * 192, *(ptr + 1) * 192);

                // Extract details for the first detected hand
                auto [sourceTriangle, keypoints] = extractHandDetails(detect, center_wo_offset);

                cv::Size originalSize(kHeight, kWidth);
                cv::Size padding(detPadHeight, detPadWidth);

                // If at least one hand is detected, proceed landmark detection
                if (!sourceTriangle.empty()){
                    // Compute scale and affine transformation matrix
                    float scale = static_cast<float>(std::max(originalSize.width, originalSize.height)) / detInputSize;
                    cv::Mat affineMatrix = computeAffineMatrix(sourceTriangle, scale);

                    // Hand landmark model inference
                    {
                        auto input_tensor = TRY(m_request.getInputTensor("input_1"));
                        auto input_buffer = input_tensor.getRawBuffer();
                        auto* land_raw = input_buffer.cast<float>();
                        cv::Mat input(kInputSize, kInputSize, CV_32FC3, land_raw);
                        warpHandRegion(land_data, affineMatrix, input);
                    }

                    auto palm_post_end = timer::now();
                    auto palm_post_time = (palm_post_end - palm_end).count();
                    CHECK(m_request.infer());
                    CHECK(m_request.wait());
                    auto landmark_end = timer::now();
                    auto landmark_time = (landmark_end - palm_post_end).count();

                    auto output_tensor = TRY(m_request.getOutputTensor("Identity"));
                    auto output_buffer = output_tensor.getRawBuffer();
                    auto* outraw = output_buffer.cast<float>();

                    // Extract landmarks
                    std::vector<std::array<float, 3>> joints = extractLandmarks(outraw);
                    
                    // Pad affine matrix and compute inverse
                    cv::Mat paddedMatrix = cv::Mat::eye(3, 3, CV_32F);
                    affineMatrix.copyTo(paddedMatrix(cv::Rect(0, 0, 3, 2)));
                    cv::Mat inverseMatrix = paddedMatrix.inv();        
                    
                    landmarks = projectLandmarksToOriginal(joints, inverseMatrix, padding);
                    handDetected = true;

                    auto end = timer::now();
                    auto landmark_post_time = (end - landmark_end).count();

                    auto total_time = palm_time + palm_post_time + landmark_time + landmark_post_time;

#if 0
                    std::cout << "Optimium Total time: " << total_time / 1000.0f << "us" << std::endl;
                    std::cout << "  - Palm: " << palm_time / 1000.0f << "us (" << palm_time * 100.0f / total_time << "%)" << std::endl;
                    std::cout << "  - Palm post: " << palm_post_time / 1000.0f << "us (" << palm_post_time * 100.0f / total_time << "%)" << std::endl;
                    std::cout << "  - Landmark: " << landmark_time / 1000.0f << "us (" << landmark_time * 100.0f / total_time << "%)" << std::endl;
                    std::cout << "  - Landmark post: " << landmark_post_time / 1000.0f << "us (" << landmark_post_time * 100.0f / total_time << "%)" << std::endl;
#endif
                    models_latencies[models_counter++ % 10] += total_time;
                    if (models_counter > 10){
                        models_average = std::accumulate(std::begin(models_latencies), std::end(models_latencies), int64_t(0)) / (10 * 1000000.0f);
                    }
                }
            }

            return rt::Ok();
        }();

        if (!result.ok()) {
            std::cerr << "failed to infer: " << result.error() << "\n";
            return false;
        }

        return handDetected;
    }

private:
    rt::Context context;
    rt::ModelOptions m_options; // for 0.3.10
    rt::Model m_model;
    rt::InferRequest m_request;
    rt::ModelOptions det_options; // for 0.3.10
    rt::Model det_model;
    rt::InferRequest det_request;

    size_t m_output_size = 0;
    float* anchors;

    bool handDetected = false;

    std::vector<BoundBox> candidateDetect;
    std::vector<float> filteredProbabilities;
    std::vector<int> indices;
    std::vector<int> boxIds;
};

// static
std::unique_ptr<InferEngine> InferEngine::create_optimium_engine() {
    auto engine = std::make_unique<OptimiumInferEngine>();

    auto result = engine->init();
    if (!result.ok()) {
        std::cerr << "failed to initalize model: " << result.error() << "\n";
        return nullptr;
    }

    return engine;
}
