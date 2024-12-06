#include "Defs.h"
#include "Postprocess.h"

// Load anchors from anchors.csv
float* loadAnchors(const std::string& filePath) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open anchors file at " << filePath << std::endl;
        return nullptr;
    }

    // Allocate memory for anchors based on detclnum
    float* anchors = new float[detclnum * 4]; // Assuming each anchor has 4 values
    size_t idx = 0;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string value;

        while (std::getline(lineStream, value, ',')) {
            if (idx < detclnum * 4) { // Ensure we don't exceed allocated memory
                anchors[idx++] = std::stof(value);
            } else {
                std::cerr << "Error: Anchors file contains more values than expected." << std::endl;
                delete[] anchors;
                return nullptr;
            }
        }
    }

    if (idx != detclnum * 4) {
        std::cerr << "Error: Anchors file contains fewer values than expected." << std::endl;
        delete[] anchors;
        return nullptr;
    }

    file.close();
    return anchors;
}

void sigmoid_score(float* raw_data){
    for(int i=0; i < detclnum; i++){
        raw_data[i] = 1.0f / (1.0f + std::exp(-raw_data[i]));
    }
}

// Decode bounding boxes using anchors
void decodeBoundingBoxes(float* decodedBoxes, const float* rawBoxes, const float* anchors) {
    std::memcpy(decodedBoxes, rawBoxes, sizeof(float) * detclnum * 18);

    for (size_t i = 0; i < detclnum; ++i) {
        const float* anchor = &anchors[i * 4];       // Each anchor has 4 values
        const float* rawBox = &rawBoxes[i * 18];    // Each raw box has 18 values
        float* decodedBox = &decodedBoxes[i * 18];  // Each decoded box also has 18 values

        // Decode center coordinates, width, and height
        decodedBox[0] = rawBox[0] + anchor[0] * 192; // dx + anchor_x * input size
        decodedBox[1] = rawBox[1] + anchor[1] * 192; // dy + anchor_y * input size
        decodedBox[2] = rawBox[2];                   // width remains the same
        decodedBox[3] = rawBox[3];                   // height remains the same

        // Copy the remaining keypoints as-is
        for (int j = 4; j < 18; ++j) {
            decodedBox[j] = rawBox[j];
        }
    }
}

// Compute the transformation triangle based on keypoints
std::vector<cv::Point2f> getTriangle(const cv::Point2f& kp0, const cv::Point2f& kp2, float side, float boxShift) {
    // Compute direction vector from wrist to middle finger keypoint
    cv::Point2f dir = kp2 - kp0;
    float length = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    dir /= length; // Normalize the direction vector

    // Compute the perpendicular vector
    cv::Point2f dirPerpendicular(dir.y, -dir.x);

    // Compute the triangle vertices
    std::vector<cv::Point2f> triangle = {
        kp2,
        kp2 + dir * side,
        kp2 + dirPerpendicular * side
    };

    // Adjust the triangle by box shift
    cv::Point2f adjustment = (kp0 - kp2) * boxShift;
    for (auto& point : triangle) {
        point -= adjustment;
    }

    return triangle;
}

// Extract keypoints and transformation triangle
std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractHandDetails(
    const BoundBox& detect,
    cv::Point2f offset
) {
    std::vector<cv::Point2f> keypoints;
    for (auto i = 0; i < 7; ++i) {
        auto idx = i * 2;
        float x = detect[4 + idx] + offset.x;
        float y = detect[4 + idx + 1] + offset.y;

        keypoints.emplace_back(x, y); 
    }

    float w = detect[2];
    float h = detect[3];
    float side = std::max(w, h) * boxEnlarge;

    auto sourceTriangle = getTriangle(keypoints[0], keypoints[2], side, boxShift);

    return {sourceTriangle, keypoints};
}

cv::Mat computeAffineMatrix(
    const std::vector<cv::Point2f>& sourceTriangle, 
    float scale
) {
    // Target triangle for the 224x224 input size
    std::vector<cv::Point2f> targetTriangle = {
        {112.0f, 112.0f}, // Center
        {112.0f, 0.0f},   // Top
        {0.0f, 112.0f}    // Left
    };

    // Scale the source triangle
    std::vector<cv::Point2f> scaledSource;
    for (const auto& pt : sourceTriangle) {
        scaledSource.emplace_back(pt.x * scale, pt.y * scale);
    }

    // Compute affine transformation matrix
    return cv::getAffineTransform(scaledSource, targetTriangle);
}

void warpHandRegion(
    const cv::Mat& imgPad, 
    const cv::Mat& affineMatrix,
    cv::Mat &output
) {
    cv::Mat imgPad_normed;
    imgPad.convertTo(imgPad_normed, CV_32F, 1.0 / 255.0);
    cv::warpAffine(imgPad_normed, output, affineMatrix, {224, 224});
}

std::vector<std::array<float, 3>> extractLandmarks(const float* outraw) {
    std::vector<std::array<float, 3>> keypoints;

    for (size_t i = 0; i < 21; ++i) {
        keypoints.push_back({outraw[i * 3], outraw[i * 3 + 1], outraw[i * 3 + 2]});
    }

    return keypoints;
}

cv::Mat padAffineMatrix(const cv::Mat& affineMatrix) {
    cv::Mat paddedMatrix = cv::Mat::eye(3, 3, CV_32F); // Create 3x3 identity matrix
    affineMatrix.copyTo(paddedMatrix(cv::Rect(0, 0, 3, 2))); // Copy affine matrix into top rows
    return paddedMatrix;
}

cv::Mat computeInverseMatrix(const cv::Mat& paddedMatrix) {
    return paddedMatrix.inv(); // Compute inverse matrix
}

std::vector<cv::Point> projectLandmarksToOriginal(
    const std::vector<std::array<float, 3>>& keypoints,
    const cv::Mat& inverseMatrix,
    const cv::Size& padding
) {
    std::vector<cv::Point> projectedKeypoints;

    auto t = inverseMatrix.t();

    for (const auto& joint : keypoints) {
        cv::Point3f homogeneousPoint(joint[0], joint[1], 1.0f); // Convert to homogeneous coordinates
        cv::Mat transformedPoint = cv::Mat(homogeneousPoint).t() * t; // Apply inverse transformation

        float x = transformedPoint.at<float>(0, 0) - padding.height;
        float y = transformedPoint.at<float>(0, 1) - padding.width;

        projectedKeypoints.emplace_back(x, y);
    }

    return projectedKeypoints;
}
