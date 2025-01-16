#include "rgb_process.h"

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
namespace fs = std::experimental::filesystem;

U32 time_print_prog_start = clock();
U32 time_print_prog_end;
U32 g_time_start;
U32 g_time_end;

G_CONFIG cfg = { 0 };

void load_cfg()
{

	cfg.isp_gain_on = 0;
	cfg.degamma_on = 0;
	cfg.awb_on = 1;
	cfg.ccm_on = 1;
	cfg.rgbgamma_on = 1;
	cfg.sharp_on = 0;

	cfg.r_gain = 1.162 * 1024;
	cfg.b_gain = 0.978 * 1024;

	float ccm_tmp[9] = {
 1.0054, - 0.0258,   0.0686,
- 0.0815,   1.0721,   0.0170,
 0.0761, - 0.0463,   0.9144



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

void rk3588_isp(RGB* rgb_data, IMG_CONTEXT context, G_CONFIG cfg)
{
	YUV* yuv_data = NULL;
	//进入RGB域

	awb_process(rgb_data, context, cfg);
	ccm_process(rgb_data, context, cfg);
	rgbgamma_process(rgb_data, context, cfg);

	yuv_data = r2y_process(rgb_data, context, cfg);
	//进入YUV域

	sharp_process(yuv_data, context, cfg);

	rgb_data = y2r_process(yuv_data, context, cfg);

	
	safe_free(yuv_data);
}

int main() 
{
	load_cfg();
	IMG_CONTEXT context = { 0 };
	const char* filename = "data/0.jpg";
	RGB* rgb_data = load_img(filename, &context);
	clear_tmp();

	save_img_with_timestamp(rgb_data, &context, "_original");
	
	degamma_process(rgb_data, context, cfg);

	rk3588_isp(rgb_data, context, cfg);


	save_img_with_timestamp(rgb_data, &context, "_final");
    // 释放内存
	safe_free(rgb_data);
	

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

RGB* load_img(const char* filename, IMG_CONTEXT* context)
{
	// 使用 OpenCV 加载图像
	cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
	if (img.empty()) {
		fprintf(stderr, "无法加载图像: %s\n", filename);
		return NULL;
	}

	// 设置图像上下文
	context->width = img.cols;
	context->height = img.rows;
	context->full_size = context->width * context->height;

	// 分配 RGB 数据的内存
	RGB* rgb_data = (RGB*)malloc(context->full_size * sizeof(RGB));
	if (!rgb_data) {
		fprintf(stderr, "RGB 数据内存分配失败.\n");
		return NULL;
	}

	// 填充 RGB 数据
	for (int y = 0; y < context->height; y++) {
		for (int x = 0; x < context->width; x++) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
			rgb_data[y * context->width + x].b = pixel[0];
			rgb_data[y * context->width + x].g = pixel[1];
			rgb_data[y * context->width + x].r = pixel[2];
		}
	}

	return rgb_data;
}


void clear_tmp() {
	std::string extensions[] = { ".jpg", ".png", ".bmp" };

	for (const auto& entry : fs::directory_iterator(".")) {
		if (fs::is_regular_file(entry.path())) { // Use fs::is_regular_file
			std::string file_ext = entry.path().extension().string();

			for (const auto& ext : extensions) {
				if (file_ext == ext) {
					try {
						fs::remove(entry.path());
						//std::cout << "删除文件: " << entry.path() << std::endl;
					}
					catch (const std::exception& e) {
						std::cerr << "删除文件失败: " << entry.path() << " 错误: " << e.what() << std::endl;
					}
					break;
				}
			}
		}
	}
}

void save_img(const char* filename, RGB* img, IMG_CONTEXT* context, int compression_quality = 100)
{
	// 创建一个空的 OpenCV Mat 对象
	cv::Mat mat_img(context->height, context->width, CV_8UC3);

	// 填充 Mat 对象
	for (int y = 0; y < context->height; y++) {
		for (int x = 0; x < context->width; x++) {
			RGB pixel = img[y * context->width + x];
			mat_img.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel.b, pixel.g, pixel.r);
		}
	}

	// 设置图像保存参数，例如压缩质量
	std::vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);  // 对 JPEG 格式指定压缩质量
	params.push_back(compression_quality);

	// 使用 OpenCV 保存图像，传递参数来设置压缩质量
	bool success = cv::imwrite(filename, mat_img, params);

	if (success) {
		LOG("%s saved (Q=%d)", filename, compression_quality);
	}
	else {
		LOG("Failed to save %s.", filename);
	}
}


void save_img_with_timestamp(RGB* rgb_data, IMG_CONTEXT* context, const char *suffix) {
	// 获取当前时间戳
	SYSTEMTIME st;
	GetSystemTime(&st);

	// 格式化时间戳
	char buffer[80];
	snprintf(buffer, sizeof(buffer), "%02d%02d%02d%03d", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);

	// 生成文件名
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s.jpg", buffer, suffix);

	// 保存 BMP 文件
	save_img(filename, rgb_data, context);
}
