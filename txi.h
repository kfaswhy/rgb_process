#pragma once

#include "rgb_process.h"

U8 txi_process(RGB* rgb, IMG_CONTEXT context, G_CONFIG cfg);

void printMatInfo(const cv::Mat& mat);

void TXI(cv::Mat src, cv::Mat dst, int rows, int cols, int size, float en, int filtertype, int filtersize);

