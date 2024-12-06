#ifndef NMS_H
#define NMS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>


struct BoundBox final {
    float *ptr;

    BoundBox(float *ptr) : ptr(ptr) {}

    float &operator [](size_t idx) { return ptr[idx]; }
    float operator [](size_t idx) const { return ptr[idx]; }
};


// Compute IoU (Intersection over Union) between two bounding boxes
float computeIoU(const BoundBox& box1, const BoundBox& box2);

// Perform Non-Maximum Suppression
std::vector<int> nonMaximumSuppression(    
    const std::vector<BoundBox>& boxes,
    const std::vector<float>& probabilities,
    int maxHands = 1
);

#endif // NMS_H
