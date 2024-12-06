#include "nms.h"
#include "Defs.h"

// Compute IoU (Intersection over Union) between two bounding boxes
float computeIoU(const BoundBox& box1, const BoundBox& box2) {
    float x1 = std::max(box1[0] - box1[2] / 2.0f, box2[0] - box2[2] / 2.0f);
    float y1 = std::max(box1[1] - box1[3] / 2.0f, box2[1] - box2[3] / 2.0f);
    float x2 = std::min(box1[0] + box1[2] / 2.0f, box2[0] + box2[2] / 2.0f);
    float y2 = std::min(box1[1] + box1[3] / 2.0f, box2[1] + box2[3] / 2.0f);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float intersection = w * h;

    float area1 = box1[2] * box1[3];
    float area2 = box2[2] * box2[3];
    float unionArea = area1 + area2 - intersection;

    return intersection / unionArea;
}

// Perform Non-Maximum Suppression
std::vector<int> nonMaximumSuppression(
    const std::vector<BoundBox>& boxes,
    const std::vector<float>& probabilities,
    int maxHands
) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0); // Initialize indices [0, 1, 2, ...]

    // Sort indices based on probabilities
    std::sort(indices.begin(), indices.end(), [&probabilities](int i, int j) {
        return probabilities[i] > probabilities[j];
    });

    int handCount = 0;
    std::vector<int> pick;
    while (!indices.empty()) {
        int current = indices.front();
        pick.push_back(current);
        handCount += 1;
        if (handCount == maxHands)
            break;
        indices.erase(indices.begin());
        indices.erase(std::remove_if(indices.begin(), indices.end(), [&](int idx) {
            float iou = computeIoU(boxes[current], boxes[idx]);
            return iou > minSuppressionThreshold;
        }), indices.end());
    }
    return pick;
}
