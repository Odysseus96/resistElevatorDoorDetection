#ifndef __COMMON_H__
#define __COMMON_H__

#include <cmath>
#include <vector>

#define OK 1
#define ERR -1

struct ObjectInfo
{
    float boxes[4];
    float conf;
    int classIdx;
};

static inline float sigmoid(float x)
{
    return static_cast<float> (1.f / (1.f + exp(-x)));
}

void QsortDescentInplace(std::vector<ObjectInfo> &detections, int left, int right);
void QsortDescentInplace(std::vector<ObjectInfo> &detections);

void NonMaxSuppression(std::vector<ObjectInfo> &detection, float nms_thresh);

float Iou(float box1[4], float box2[4]);


#endif //__COMMON_H__