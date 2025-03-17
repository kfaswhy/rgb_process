#pragma once

#include "rgb_process.h"

RGB* interp_with_optical_flow(RGB* rgb_data0, RGB* rgb_data1, IMG_CONTEXT context, float alpha);