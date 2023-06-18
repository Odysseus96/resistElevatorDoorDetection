#include "common.h"
#include "loguru.hpp"
#include <algorithm>


void QsortDescentInplace(std::vector<ObjectInfo> &detections, int left, int right)
{
    int i = left;
    int j = right;

    float p = detections[(left + right) / 2].conf;

    while (i <= j)
    {
        while (detections[i].conf > p)
            i++;

        while (detections[i].conf < p)
            j--;
        
        if (i <= j)
        {
            std::swap(detections[i], detections[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) QsortDescentInplace(detections, left, j);
        }

        #pragma omp section
        {
            if (right > i) QsortDescentInplace(detections, i, right);
        }
    }
}

void QsortDescentInplace(std::vector<ObjectInfo> &detections)
{
    if (detections.empty())
    {
        LOG_F(INFO, "没有检测到有效目标");
        return;
    }
    QsortDescentInplace(detections, 0, detections.size() - 1);
}


float Iou(float box1[4], float box2[4]) // box x, y, w, h
{
    float interBox[4] = {
        std::max((box1[0] - box1[2] / 2), (box2[0] - box2[2] / 2)), // left
        std::min((box1[0] + box1[2] / 2), (box2[0] + box2[2] / 2)), // right
        std::max((box1[1] - box1[3] / 2), (box2[1] - box2[3] / 2)), // top
        std::min((box1[1] + box1[3] / 2), (box2[1] + box2[3] / 2)) // bottom
    };

    float interArea = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionArea = box1[2] * box1[3] + box1[2] * box2[3] - interArea;
    return (float)(interArea / unionArea);

}
void NonMaxSuppression(std::vector<ObjectInfo> &detection, float nms_thresh)
{
    const int nums = detection.size();
    for (int i = 0; i < nums; i++)
    {
        if (detection[i].conf < 0.1)
            continue;
        
        for (int j = i + 1; j < nums; j++)
        {
            if (Iou(detection[i].boxes, detection[j].boxes) > nms_thresh)
            {
                detection[i].conf = 0.f;
            }
        }
    }

    auto iter = detection.begin();
    while (iter != detection.end())
    {
        if (iter->conf < 0.1)
        {
            iter = detection.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
}