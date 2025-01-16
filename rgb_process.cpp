﻿#include "rgb_process.h"

#include "ob.h"
#include "isp_gain.h"
#include "awb.h"
#include "ltm.h"
#include "demosaic.h"
#include "ccm.h"
#include "rgbgamma.h"
#include "degamma.h"
#include "r2y.h"
#include "ygamma.h"
#include "sharp.h"
#include "y2r.h"

using namespace std;
using namespace cv;


U32 time_print_prog_start = clock();
U32 time_print_prog_end;
U32 g_time_start;
U32 g_time_end;

G_CONFIG cfg = { 0 };

void load_cfg()
{
	//cfg.bit = 16;
	//cfg.used_bit = 12;
	//cfg.order = LITTLE_ENDIAN;
	//cfg.pattern = RGGB;
	//cfg.width = 2592;
	//cfg.height = 1536;
	//以上没用了
	
	//cfg.ob_on = 1;
	cfg.isp_gain_on = 0;

	cfg.degamma_on = 0;
	cfg.awb_on = 1;
	//cfg.ltm_on = 1;
	cfg.ccm_on = 1;
	cfg.rgbgamma_on = 1;
	
	//cfg.ygamma_on = 0;
	cfg.sharp_on = 0;

	//cfg.ob = 4096 ;
	//cfg.isp_gain = 1524;

	cfg.r_gain = 1.15 *1024;
	cfg.b_gain = 1.2222222 *1024;

	//cfg.ltm_strength = 0.2;
	//cfg.ltm_vblk = 4;
	//cfg.ltm_hblk = 2;
	//cfg.ltm_cst_thdr = 1;

	float ccm_tmp[9] = {
1.1787 ,  0.0665 , -0.3283  ,
-0.3505 ,  1.7091 , -0.5077  ,
 0.20 , -0.7756  , 1.8360



	};


	U32 gamma_xtmp[49] =
	{
		0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4095
	};

	U32 gamma_ytmp[49] =
	{
		0,6,11,17,22,28,33,39,44,55,66,77,88,109,130,150,170,210,248,286,323,393,460,525,586,702,809,909,1002,1172,1323,1461,1587,1810,2003,2173,2325,2589,2812,3010,3191,3355,3499,3624,3736,3836,3927,4012,4095
	};

	U32 degamma_xtmp[49] =
	{
		0, 6, 11, 17, 22, 28, 33, 39, 44, 55, 66, 77, 88, 109, 130, 150, 170, 210, 248, 286, 323, 393, 460, 525, 586, 702, 809, 909, 1002, 1172, 1323, 1461, 1587, 1810, 2003, 2173, 2325, 2589, 2812, 3010, 3191, 3355, 3499, 3624, 3736, 3836, 3927, 4012, 4095
	};

	U32 degamma_ytmp[49] =
	{ 
		0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4095
	};

	memcpy(cfg.ccm, ccm_tmp, 9 * sizeof(float));
	
	memcpy(cfg.gamma_x, gamma_xtmp, 49 * sizeof(U32));
	memcpy(cfg.gamma_y, gamma_ytmp, 49 * sizeof(U32));
	memcpy(cfg.degamma_x, degamma_xtmp, 49 * sizeof(U32));
	memcpy(cfg.degamma_y, degamma_ytmp, 49 * sizeof(U32));
	return;
}

int main() 
{
	load_cfg();
	IMG_CONTEXT context = { 0 };
	const char* filename = "data/0.bmp";
	RGB* rgb_data = load_bmp(filename, &context);
	YUV* yuv_data = NULL;

	save_bmp("0.bmp", rgb_data, &context);

	//进入RGB域

	degamma_process(rgb_data, context, cfg);
	save_bmp("1_degamma.bmp", rgb_data, &context);

	awb_process(rgb_data, context, cfg);
	save_bmp("2_awb.bmp", rgb_data, &context);

	ccm_process(rgb_data, context, cfg);
	save_bmp("3_ccm.bmp", rgb_data, &context);

	


	rgbgamma_process(rgb_data, context, cfg);
	save_bmp("4_rgbgamma.bmp", rgb_data, &context);


	yuv_data = r2y_process(rgb_data, context, cfg);
	//进入YUV域

	sharp_process(yuv_data, context, cfg);
	rgb_data = y2r_process(yuv_data, context, cfg);
	save_bmp("20_sharp.bmp", rgb_data, &context);


	rgb_data = y2r_process(yuv_data, context, cfg);
	save_bmp("99_final.bmp", rgb_data, &context);

    // 释放内存
	free(rgb_data);
	free(yuv_data);

	time_print_prog_end = clock();
	LOG("time = %.2f s.", ((float)time_print_prog_end - time_print_prog_start) / 1000);


    return 0;
}

U32 calc_inter(U32 x0, U32* x, U32* y, U32 len)
{
	U32 y0 = 0;

	if (len < 2) {
		// 长度小于2无法插值
		return y0;
	}

	// 判断递增还是递减
	int increasing = (x[1] > x[0]) ? 1 : 0;

	// 寻找x0所在的区间
	for (U32 i = 0; i < len - 1; i++) {
		if ((increasing && x0 >= x[i] && x0 <= x[i + 1]) ||
			(!increasing && x0 <= x[i] && x0 >= x[i + 1])) {
			// 线性插值计算y0
			U32 x1 = x[i], x2 = x[i + 1];
			U32 y1 = y[i], y2 = y[i + 1];

			y0 = y1 + (y2 - y1) * (x0 - x1) / (x2 - x1);
			return y0;
		}
	}

	// 如果x0不在范围内，返回边界值
	if (increasing) {
		y0 = (x0 < x[0]) ? y[0] : y[len - 1];
	}
	else {
		y0 = (x0 > x[0]) ? y[0] : y[len - 1];
	}

	return y0;
}

RGB* raw2rgb(U16* raw, IMG_CONTEXT context, G_CONFIG cfg) {
    // 获取图像的宽高和 Bayer Pattern
    U16 width = context.width;
    U16 height = context.height;
    BayerPattern pattern = (BayerPattern)cfg.pattern;
    ByteOrder order = (ByteOrder)cfg.order;
    const U8 bit_depth = cfg.bit;
    const int bit_shift = cfg.bit - 8;

    // 确定最大有效值
    U16 max_val = (1 << bit_depth) - 1;

    // 确保原始数据按大小端顺序处理
    for (U32 i = 0; i < width * height; i++) {
        if (order == LITTLE_ENDIAN) {
            raw[i] = raw[i] & max_val; // 提取低 bit_depth 位
        }
        else if (order == BIG_ENDIAN) {
            raw[i] = ((raw[i] & 0xFF) << 8 | (raw[i] >> 8)) & max_val;
        }
    }

    // 分配 RGB 数据的内存
    RGB* rgb_data = (RGB*)malloc(width * height * sizeof(RGB));
    if (!rgb_data) {
        fprintf(stderr, "Memory allocation for RGB data failed.\n");
        return NULL;
    }

    // 根据 Bayer Pattern 进行插值处理
    for (U16 y = 0; y < height; y++) {
        for (U16 x = 0; x < width; x++) {
            RGB pixel = { 0, 0, 0 };
            U32 val = 0;

            // 插值计算
            switch (pattern) {
            case RGGB:
                if ((y % 2 == 0) && (x % 2 == 0)) //R
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.r = clp_range(0, val, U8MAX);
					pixel.g = 0;
					pixel.b = 0;
                }
                else if ((y % 2 == 0) && (x % 2 == 1)) //GR
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);
                    pixel.r = 0;
                    pixel.b = 0;
                }
                else if ((y % 2 == 1) && (x % 2 == 0)) //GB
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.g = clp_range(0, val, U8MAX);
					pixel.r = 0;
					pixel.b = 0;
                }
                else //B
                {
                    val = raw[y * width + x] >> bit_shift;
                    pixel.b = clp_range(0, val, U8MAX);
					pixel.r = 0;
					pixel.g = 0;
                }
				break;
            case BGGR:
                if ((y % 2 == 0) && (x % 2 == 0)) //B
                {
					val = raw[y * width + x] >> bit_shift;
					pixel.b = clp_range(0, val, U8MAX);
					pixel.r = 0;
					pixel.g = 0;
                }
                else if ((y % 2 == 0) && (x % 2 == 1)) //GB
                {
					val = raw[y * width + x] >> bit_shift;
					pixel.g = clp_range(0, val, U8MAX);
					pixel.r = 0;
					pixel.b = 0;
                }
                else if ((y % 2 == 1) && (x % 2 == 0)) //GR
                {
					val = raw[y * width + x] >> bit_shift;
					pixel.g = clp_range(0, val, U8MAX);
					pixel.r = 0;
					pixel.b = 0;
                }
                else //R
                {
					val = raw[y * width + x] >> bit_shift;
					pixel.r = clp_range(0, val, U8MAX);
					pixel.g = 0;
					pixel.b = 0;
                }
                break;
                // Other patterns (GRBG, GBRG, BGGR) can be implemented similarly
            default:
                fprintf(stderr, "Unsupported Bayer Pattern.\n");
                free(rgb_data);
                return NULL;
            }

            // 保存到 RGB 数据
            rgb_data[y * width + x] = pixel;
        }
    }

    return rgb_data;
}





U16* readraw(const char* filename, IMG_CONTEXT context, G_CONFIG cfg)
{
    int bytesPerPixel = (cfg.bit + 7) / 8;  // 计算每像素的字节数
    int dataSize = context.width * context.height * bytesPerPixel;  // 数据总大小
    U16* raw = (U16*)malloc(context.width * context.height * sizeof(U16));

    FILE* rawFile = fopen(filename, "rb");
    if (!rawFile) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    U8* buffer = (U8*)malloc(dataSize);
    fread(buffer, 1, dataSize, rawFile);
    fclose(rawFile);

    for (int i = 0; i < context.width * context.height; i++) {
		if (cfg.bit == 16) {
			raw[i] = cfg.order == LITTLE_ENDIAN
                ? buffer[i * 2] | (buffer[i * 2 + 1] << 8)
                : (buffer[i * 2] << 8) | buffer[i * 2 + 1];
        }
        else {
			raw[i] = buffer[i];
        }

		raw[i] = raw[i] << (cfg.bit - cfg.used_bit);
    }

    free(buffer);
    return raw;
}

U8 save_rgb(const char* filename, RGB* rgb_data, IMG_CONTEXT context, G_CONFIG cfg) {
	// 创建 OpenCV Mat 对象
	int width = context.width;
	int height = context.height;
	cv::Mat img(height, width, CV_8UC3);

	// 填充 Mat 数据
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			RGB pixel = rgb_data[y * width + x];
			img.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel.b, pixel.g, pixel.r);
		}
	}

	// 保存到文件
	if (cv::imwrite(filename, img)) {
		return 1; // 保存成功
	}
	else {
		return 0; // 保存失败
	}
}

void safe_free(void* p)
{
	if (NULL != p)
	{
		free(p);
		p = NULL;
	}
	return;
}

void print_prog(U32 cur_pos, U32 tgt)
{
	time_print_prog_end = clock();
	if ((time_print_prog_end - time_print_prog_start) >= 1000)
	{
		LOG("Processing: %d%%.", cur_pos * 100 / tgt);
		time_print_prog_start = clock();
	}
	return;
}

RGB* load_bmp(const char* filename, IMG_CONTEXT* context)
{
	FILE* f_in = fopen(filename, "rb");
	if (f_in == NULL)
	{
		LOG("Cannot find %s", filename);
		return NULL;
	}

	fread(&context->fileHeader, sizeof(BITMAPFILEHEADER), 1, f_in);
	fread(&context->infoHeader, sizeof(BITMAPINFOHEADER), 1, f_in);

	context->height = context->infoHeader.biHeight;
	context->width = context->infoHeader.biWidth;
	context->full_size = context->width * context->height;
	//context->w_samp = context->width / cfg.sample_ratio;
	//context->h_samp = context->height / cfg.sample_ratio;


	int LineByteCnt = (((context->width * context->infoHeader.biBitCount) + 31) >> 5) << 2;
	//int ImageDataSize = LineByteCnt * height;
	context->PaddingSize = 4 - ((context->width * context->infoHeader.biBitCount) >> 3) & 3;
	context->pad = (BYTE*)malloc(sizeof(BYTE) * context->PaddingSize);
	RGB* img = (RGB*)malloc(sizeof(RGB) * context->height * context->width);

	if (context->infoHeader.biBitCount == 24) {
		for (int i = 0; i < context->height; i++) {
			for (int j = 0; j < context->width; j++) {
				int index = i * context->width + j;
				fread(&img[index], sizeof(RGB), 1, f_in);
			}
			if (context->PaddingSize != 0)
			{
				fread(context->pad, 1, context->PaddingSize, f_in);
			}
		}
	}
	else if(context->infoHeader.biBitCount == 32)
    {
		for (int i = 0; i < context->height; i++) {
			for (int j = 0; j < context->width; j++) {
				int index = i * context->width + j;
				RGBQUAD quad;
				fread(&quad, sizeof(RGBQUAD), 1, f_in);
				img[index].r = quad.rgbRed;
				img[index].g = quad.rgbGreen;
				img[index].b = quad.rgbBlue;
			}
			if (context->PaddingSize != 0)
			{
				fread(context->pad, 1, context->PaddingSize, f_in);
			}
		}
	}
	else
	{
		LOG("Only support BMP in 24-bit.");
		return NULL;
	}

	fclose(f_in);
	return img;
}

void save_bmp(const char* filename, RGB* img, IMG_CONTEXT* context)
{
	FILE* f_out = fopen(filename, "wb");

	context->fileHeader.bfType = 0x4D42; // 'BM'
	context->fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	context->fileHeader.bfSize = context->fileHeader.bfOffBits + context->width * context->height * sizeof(RGB);
	context->fileHeader.bfReserved1 = 0;
	context->fileHeader.bfReserved2 = 0;
	 
	context->infoHeader.biSize = sizeof(BITMAPINFOHEADER);
	context->infoHeader.biWidth = context->width;
	context->infoHeader.biHeight = context->height;
	context->infoHeader.biPlanes = 1;
	context->infoHeader.biBitCount = 24;
	context->infoHeader.biCompression = 0;
	context->infoHeader.biSizeImage = context->width * context->height * sizeof(RGB);
	context->infoHeader.biXPelsPerMeter = 0;
	context->infoHeader.biYPelsPerMeter = 0;
	context->infoHeader.biClrUsed = 0;
	context->infoHeader.biClrImportant = 0;

	fwrite(&context->fileHeader, sizeof(context->fileHeader), 1, f_out);
	fwrite(&context->infoHeader, sizeof(context->infoHeader), 1, f_out);

	for (int i = 0; i < context->height; i++)
	{
		for (int j = 0; j < context->width; j++)
		{
			fwrite(&img[i * context->width + j], sizeof(RGB), 1, f_out);
		}
		if (context->PaddingSize != 0)
		{
			fwrite(context->pad, 1, context->PaddingSize, f_out);
		}
	}


	fclose(f_out);
	
	LOG("saved %s.", filename);
	return;
}