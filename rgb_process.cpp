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
#include "txi.h"
#include "ytxi.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

U32 time_print_prog_start = clock();
U32 time_print_prog_end;
U32 g_time_start;
U32 g_time_end;

G_CONFIG cfg = { 0 };

void load_cfg()
{

	cfg.isp_gain_on = 0;
	cfg.degamma_on = 1;
	cfg.awb_on = 0;
	cfg.ccm_on = 1;
	cfg.rgbgamma_on = 1;
	cfg.sharp_on = 0;

	cfg.isp_gain = 1024 * 1.8;

	cfg.r_gain = 1024 * 1.28;
	cfg.b_gain = 1024 * 2.84;

	float ccm_tmp[9] = {
0.00, 	0.98 ,	0.16,
0.00 ,- 0.01 ,	0.77,
0.00 ,- 0.03, 	0.36




	};

	U32 gamma_xtmp[49] =
	{
		0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4095
	};

	U32 gamma_ytmp[49] =
	{
0, 6, 11, 17, 22, 28, 33, 39, 44, 55, 66, 77, 88, 109, 130, 150, 170,
210, 248, 286, 323, 393, 460, 525, 586, 702, 809, 909, 1002, 1172, 1323,
1461, 1587, 1810, 2003, 2173, 2325, 2589, 2812, 3010, 3191, 3355, 3499,
3624, 3736, 3836, 3927, 4012, 4095
	};

	U32 degamma_xtmp[49] =
	{
		//0, 63, 90, 110, 127, 143, 156, 169, 180, 202, 221, 239, 255, 286, 313, 338, 361, 404, 443, 478, 511, 572, 626, 677, 723, 809, 886, 957, 1023, 1144, 1253, 1354, 1447, 1618, 1773, 1915, 2047, 2289, 2507, 2708, 2895, 3071, 3237, 3395, 3546, 3691, 3830, 3964, 4095
		0, 6, 11, 17, 22, 28, 33, 39, 44, 55, 66, 77, 88, 109, 130, 150, 170, 210, 248, 286,		323, 393, 460, 525, 586, 702, 809, 909, 1002, 1172, 1323, 1461, 1587, 1810, 2003, 2173,		2325, 2589, 2812, 3010, 3191, 3355, 3499, 3624, 3736, 3836, 3927, 4012, 4095

		//0, 15, 25, 33, 40, 46, 52, 58, 63, 74, 83, 92, 101, 117, 133, 147, 161, 187, 211, 234, 255, 296, 335, 371, 406, 471, 532, 589, 644, 748, 845, 936, 1023, 1187, 1341, 1486, 1625, 1885, 2129, 2359, 2579, 2790, 2993, 3189, 3380, 3565, 3746, 3922, 4095

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

	isp_gain_process(rgb_data, context, cfg);
	awb_process(rgb_data, context, cfg);
	ccm_process(rgb_data, context, cfg);
	rgbgamma_process(rgb_data, context, cfg);


	yuv_data = r2y_process(rgb_data, context, cfg);
	//进入YUV域

	sharp_process(yuv_data, context, cfg);

	RGB* rgb_tmp = y2r_process(yuv_data, context, cfg);
	memcpy(rgb_data, rgb_tmp, context.full_size * sizeof(RGB));
	
	safe_free(yuv_data);
}

// 原始的增强函数，调用 screenshot_enhance_process 函数
void screenshot_enhance(RGB* rgb_data, IMG_CONTEXT context) {
	// 将 RGB 数据转换为 OpenCV 的 Mat 对象
	cv::Mat image(context.height, context.width, CV_8UC3);
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			image.at<cv::Vec3b>(y, x)[0] = rgb_data[index].b;
			image.at<cv::Vec3b>(y, x)[1] = rgb_data[index].g;
			image.at<cv::Vec3b>(y, x)[2] = rgb_data[index].r;
		}
	}

	// 中值滤波
	cv::Mat median_filtered;
	cv::medianBlur(image, median_filtered, 5);

	float sharpen_strength = 1;
	// 锐化操作，根据 sharpen_strength 调整锐化核
	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -sharpen_strength, 0, -sharpen_strength, 1 + 4 * sharpen_strength, -sharpen_strength, 0, -sharpen_strength, 0);
	cv::Mat sharpened;
	cv::filter2D(median_filtered, sharpened, -1, kernel);

	cv::Mat blended;
	float blend_ratio = 0.6;
	cv::addWeighted(image, 1 - blend_ratio, sharpened, blend_ratio, 0, blended);

	// 局部对比度提升：自适应直方图均衡化
	cv::Mat lab_image;
	cv::cvtColor(blended, lab_image, cv::COLOR_BGR2Lab);
	std::vector<cv::Mat> lab_planes(3);
	cv::split(lab_image, lab_planes);

	float clahe_strength = 0.3;

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(clahe_strength);
	clahe->apply(lab_planes[0], lab_planes[0]);

	cv::merge(lab_planes, lab_image);
	cv::cvtColor(lab_image, blended, cv::COLOR_Lab2BGR);

	// 将最终处理后的图像数据复制回输入的 Mat 对象
	blended.copyTo(image);

	// 将处理后的图像数据复制回 rgb_data
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			rgb_data[index].b = image.at<cv::Vec3b>(y, x)[0];
			rgb_data[index].g = image.at<cv::Vec3b>(y, x)[1];
			rgb_data[index].r = image.at<cv::Vec3b>(y, x)[2];
		}
	}
	save_img_with_timestamp(rgb_data, &context, "_enhance");
}

void screenshot_txi(RGB* rgb_data, IMG_CONTEXT context)
{

	//RGB域TXI
	//txi_process(rgb_data, context, cfg);

	YUV* yuv_data = r2y_process(rgb_data, context, cfg);

	txi_nv12_process(yuv_data, context, cfg);
	//ytxi_process(yuv_data, context, cfg);

	RGB* rgb_tmp = y2r_process(yuv_data, context, cfg);
	memcpy(rgb_data, rgb_tmp, context.full_size * sizeof(RGB));

	//save_img_with_timestamp(rgb_data, &context, "_ytxi");
	safe_free(yuv_data);

}

void calc_shading(RGB* rgb_data, IMG_CONTEXT context) {
	if (rgb_data == NULL) {
		printf("rgb_data is NULL\n");
		return;
	}

	int row = context.height / 2;
	FILE* fp = fopen("middle_line_rgb_y.csv", "w");
	if (!fp) {
		printf("Failed to open output file\n");
		return;
	}

	fprintf(fp, "R,G,B,Y\n");

	for (int col = 0; col < context.width; ++col) {
		int index = row * context.width + col;
		RGB pixel = rgb_data[index];

		// BT.601 standard Y = 0.299 * R + 0.587 * G + 0.114 * B
		U8 y = (U8)(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);

		fprintf(fp, "%d,%d,%d,%d\n", pixel.r, pixel.g, pixel.b, y);
	}

	fclose(fp);
	printf("CSV saved: middle_line_rgb_y.csv\n");
}

int main() 
{
	load_cfg();
	IMG_CONTEXT context = { 0 };
	RGB* rgb_data = load_img("data/0.bmp", &context);

	clear_tmp();

	save_img_with_timestamp(rgb_data, &context, "_original");
	
	
	degamma_process(rgb_data, context, cfg);
	//rk3588_isp(rgb_data, context, cfg);
	//screenshot_enhance(rgb_data, context);
	//screenshot_txi(rgb_data, context);
	//calc_shading(rgb_data,context);

	ccm_process(rgb_data, context, cfg);
	rgbgamma_process(rgb_data, context, cfg);


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
