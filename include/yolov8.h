#ifndef __YOLOV8_H__
#define __YOLOV8_H__

#include <stdint.h>
#include <string>

struct modelInput
{
    int width;
    int height;
};

class Yolov8
{
public:
    Yolov8();
    ~Yolov8();
    int32_t Init(std::string &modelPath);

    int32_t PreProcess();
    int32_t Inference();
    int32_t PostProcess();

    int32_t GenerateProposals();

private:
    float m_confidenceThres;
    float m_nmsThres;
    
    std::string m_modelPath;
};


#endif // __YOLOV8_H__