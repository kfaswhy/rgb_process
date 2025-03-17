#include "interp.h"

//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>

using namespace std;
using namespace cv;

RGB* interp_with_optical_flow(RGB* prev, RGB* next, IMG_CONTEXT context, float alpha)
{
    // 生成插值结果图像
    RGB* cur = (RGB*)calloc(context.height * context.width, sizeof(RGB));

    // 将 prev 和 next 图像转换为 cv::Mat 格式
    cv::Mat prevMat(context.height, context.width, CV_8UC3, prev);
    cv::Mat nextMat(context.height, context.width, CV_8UC3, next);

    // 分别提取 R、G 和 B 通道
    std::vector<cv::Mat> prevChannels(3), nextChannels(3);
    cv::split(prevMat, prevChannels);
    cv::split(nextMat, nextChannels);

    // 为每个通道计算光流
    cv::Mat flowR, flowG, flowB;
    cv::Mat prevGray, nextGray;

    // 将每个通道转换为灰度图像（单通道）
    cv::cvtColor(prevMat, prevGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(nextMat, nextGray, cv::COLOR_BGR2GRAY);

    // 计算 R、G 和 B 通道的光流
    cv::calcOpticalFlowFarneback(prevChannels[2], nextChannels[2], flowR, 0.5, 3, 15, 3, 5, 1.2, 0);  // Red channel
    cv::calcOpticalFlowFarneback(prevChannels[1], nextChannels[1], flowG, 0.5, 3, 15, 3, 5, 1.2, 0);  // Green channel
    cv::calcOpticalFlowFarneback(prevChannels[0], nextChannels[0], flowB, 0.5, 3, 15, 3, 5, 1.2, 0);  // Blue channel

    // 使用光流进行插值
    for (int y = 0; y < context.height; y++) {
        for (int x = 0; x < context.width; x++) {
            // 计算每个通道的光流
            cv::Point2f flowAtPointR = flowR.at<cv::Point2f>(y, x);
            cv::Point2f flowAtPointG = flowG.at<cv::Point2f>(y, x);
            cv::Point2f flowAtPointB = flowB.at<cv::Point2f>(y, x);

            // 插值位置
            float newX = x + flowAtPointR.x * alpha;
            float newY = y + flowAtPointR.y * alpha;

            // 保证新位置在图像边界内
            newX = std::max(0.0f, std::min((float)context.width - 1, newX));
            newY = std::max(0.0f, std::min((float)context.height - 1, newY));

            // 获取每个通道的插值像素
            int x0 = (int)newX;
            int y0 = (int)newY;
            int x1 = std::min(x0 + 1, context.width - 1);
            int y1 = std::min(y0 + 1, context.height - 1);

            // 双线性插值，分别处理每个通道
            RGB* prevPixelR = &prev[y0 * context.width + x0];
            RGB* nextPixelR = &next[y0 * context.width + x0];

            RGB* prevPixelG = &prev[y0 * context.width + x0];
            RGB* nextPixelG = &next[y0 * context.width + x0];

            RGB* prevPixelB = &prev[y0 * context.width + x0];
            RGB* nextPixelB = &next[y0 * context.width + x0];

            RGB* curPixel = &cur[y * context.width + x];

            // 根据 alpha 做插值
            curPixel->r = (U8)(prevPixelR->r * (1 - alpha) + nextPixelR->r * alpha);
            curPixel->g = (U8)(prevPixelG->g * (1 - alpha) + nextPixelG->g * alpha);
            curPixel->b = (U8)(prevPixelB->b * (1 - alpha) + nextPixelB->b * alpha);
        }
    }

    return cur;
}