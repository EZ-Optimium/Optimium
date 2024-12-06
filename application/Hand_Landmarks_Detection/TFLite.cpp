#include "InferEngine.h"
#include "Defs.h"
#include "Postprocess.h"
#include "nms.h"

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <tensorflow/lite/model.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <cmath>
#include <numeric>
#include <iterator>
#include <iostream>

constexpr auto TFLiteDetModelPath = "palm_detection_lite.tflite";
constexpr auto TFLiteLandmarkModelPath = "hand_landmark_lite.tflite";
constexpr auto AnchorsPath = "anchors.csv";

using timer = std::chrono::high_resolution_clock;

class TFLiteInferEngine final : public InferEngine {
public:
    TFLiteInferEngine(std::unique_ptr<tflite::Interpreter> interpreter, std::unique_ptr<tflite::Interpreter> detinterpreter)
        : m_interpreter(std::move(interpreter)), det_interpreter(std::move(detinterpreter)) {
        m_output_size = m_interpreter->output_tensor(0)->bytes / sizeof(float);
        det_output_size = det_interpreter->output_tensor(0)->bytes / sizeof(float); 
        anchors = loadAnchors(AnchorsPath);
    }

    bool do_infer(const cv::Mat& data, cv::Mat& land_data, std::vector<cv::Point>& landmarks, std::vector<cv::Rect>& faces) override {
        handDetected = false;
        auto* raw_input = det_interpreter->typed_input_tensor<float>(0);
        
        cv::Mat det_input(detInputSize, detInputSize, CV_32FC3, raw_input);
        data.copyTo(det_input);
	
        auto begin = timer::now();
        if (det_interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "error: failed to invoke interpreter.\n";
            return false;
        }
        auto palm_end = timer::now();
        auto palm_time = (palm_end - begin).count();
        
        auto* rawBoxes = det_interpreter->typed_output_tensor<float>(0);
        auto* rawScores = det_interpreter->typed_output_tensor<float>(1);

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

        // If at least one box is detected, proceed palm(hand) detection
        boxIds.clear();
        boxIds = nonMaximumSuppression(candidateDetect, filteredProbabilities);

        // If at least single box is detected, rpoceed landmark detection
        if (!boxIds.empty()){
            int boxId = boxIds[0];
            BoundBox detect = candidateDetect[boxId];
            auto *ptr = anchors + 4 * indices[boxId];
            cv::Point2f center_wo_offset(*ptr * 192, *(ptr + 1) * 192);

            // Extract details for the first detected hand
            auto [sourceTriangle, keypoints] = extractHandDetails(
                detect, center_wo_offset
            );

            cv::Size originalSize(kHeight, kWidth);
            cv::Size padding(detPadHeight, detPadWidth);

            // If at least one hand is detected, proceed landmark detection
            if (!sourceTriangle.empty()){
                // Compute scale and affine transformation matrix
                float scale = static_cast<float>(std::max(originalSize.width, originalSize.height)) / detInputSize;
                cv::Mat affineMatrix = computeAffineMatrix(sourceTriangle, scale);

                // Warp the hand region
                // cv::Mat warpedImage = warpHandRegion(land_data, affineMatrix);

                // Hand landmark model inference
                auto* landmark_raw_input = m_interpreter->typed_input_tensor<float>(0);
                cv::Mat landmark_input(kInputSize, kInputSize, CV_32FC3, landmark_raw_input);
                warpHandRegion(land_data, affineMatrix, landmark_input);

                auto palm_post_end = timer::now();
                auto palm_post_time = (palm_post_end - palm_end).count();
                if (m_interpreter->Invoke() != kTfLiteOk) {
                    std::cerr << "error: failed to invoke interpreter.\n";
                    return false;
                }
                auto landmark_end = timer::now();
                auto landmark_time = (landmark_end - palm_post_end).count();

                auto* outraw = m_interpreter->typed_output_tensor<float>(0);

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
                std::cout << "TFLite Total time: " << total_time / 1000.0f << "us" << std::endl;
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
        return handDetected;
    }

private:
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    std::unique_ptr<tflite::Interpreter> det_interpreter;
    size_t m_output_size;
    size_t det_output_size;
    float* anchors;

    bool handDetected = false;

    std::vector<BoundBox> candidateDetect;
    std::vector<float> filteredProbabilities;
    std::vector<int> indices;
    std::vector<int> boxIds;
};

// static
std::unique_ptr<InferEngine> InferEngine::create_tflite_engine() {
    auto detmodel = tflite::FlatBufferModel::BuildFromFile(TFLiteDetModelPath);
    auto model = tflite::FlatBufferModel::BuildFromFile(TFLiteLandmarkModelPath);

    if (!detmodel) {
        std::cerr << "error: failed to tflite detection model\n";
        return nullptr;
    }

    if (!model) {
        std::cerr << "error: failed to tflite model\n";
        return nullptr;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> detinterpreter;
    auto detbuilder = tflite::InterpreterBuilder(*detmodel, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    auto builder = tflite::InterpreterBuilder(*model, resolver);

    if (detbuilder(&detinterpreter) != TfLiteStatus::kTfLiteOk) {
        std::cerr << "error: failed to create det interpreter.\n";
        return nullptr;
    }

    if (builder(&interpreter) != TfLiteStatus::kTfLiteOk) {
        std::cerr << "error: failed to create interpreter.\n";
        return nullptr;
    }

    detbuilder.SetNumThreads(2);
    builder.SetNumThreads(2);
    interpreter->SetNumThreads(2);
    detinterpreter->SetNumThreads(2);

    detinterpreter->SetAllowFp16PrecisionForFp32(false);
    interpreter->SetAllowFp16PrecisionForFp32(false);

    if (detinterpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "error: failed to allocate det tensors.\n";
        return nullptr;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "error: failed to allocate tensors.\n";
        return nullptr;
    }

    return std::make_unique<TFLiteInferEngine>(std::move(interpreter), std::move(detinterpreter));
}