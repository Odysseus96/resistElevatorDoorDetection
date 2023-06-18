#include "common.h"
#include "yolov8.h"

#include <vector>
#include <loguru.hpp>

int main(int argc, char* argv[])
{
    std::shared_ptr<Yolov8> yolov8 = std::make_shared<Yolov8>();
    
    loguru::init(argc, argv);
    loguru::add_file("Yolov8Detection.log", loguru::Append, loguru::Verbosity_MAX);

    std::string modelPath = "models/elevator-close-detv8s.onnx";

    if (yolov8->Init(modelPath) != OK)
    {
        LOG_F(INFO, "Yolov8 初始化失败");
        return ERR;
    }
    LOG_F(INFO, "Yolov8 初始化成功");

    return 0;
}