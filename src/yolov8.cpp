#include <stdint.h>
#include <openvino/openvino.hpp>
#include "yolov8.h"
#include "loguru.hpp"
#include "common.h"

Yolov8::Yolov8()
{
    LOG_F(INFO, "Yolov8 构建成功");
}

Yolov8::~Yolov8()
{
    LOG_F(INFO, "Yolov8 销毁成功");
}

int32_t Yolov8::Init(std::string &modelPath)
{
    this->m_confidenceThres = 0.45;
    this->m_modelPath = modelPath;
    return OK;
}