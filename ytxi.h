#pragma once

#include "rgb_process.h"

void save_mat(const cv::Mat& img, const std::string& filename);

void YTXI(cv::Mat src, cv::Mat dst, int rows, int cols, int size, float en, int filtertype, int filtersize);

U8 ytxi_process(YUV* yuv, IMG_CONTEXT context, G_CONFIG cfg);

U8 txi_nv12_process(YUV* yuv, IMG_CONTEXT context, G_CONFIG cfg);

