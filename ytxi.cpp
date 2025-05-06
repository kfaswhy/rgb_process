#include "ytxi.h"
#include "y2r.h"
using namespace std;
using namespace cv;

void save_mat(const cv::Mat& img, const std::string& filename)
{
#if DEBUG_MODE
	cv::Mat normalized_img;
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);

	if (minVal < 0) {

		cv::Mat exp_img;
		exp(img, exp_img);
		exp_img.convertTo(normalized_img, CV_8U, 255.0);
	}
	else
		if (img.depth() != CV_8U) {

			img.convertTo(normalized_img, CV_8U, 255.0);
		}
		else {
			normalized_img = img.clone();
		}

	// 保存图像为 JPEG 格式
	bool saved = cv::imwrite(filename, normalized_img);
	if (!saved) {
		std::cerr << "Failed to save the image to " << filename << std::endl;
	}
#endif
}

void YTXI(cv::Mat src, cv::Mat dst, int rows, int cols, int size, float en, int filtertype, int filtersize)
{
    // 提取第一个通道
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    cv::Mat first_channel = channels[0];

    cv::Mat src_f(rows, cols, CV_32FC1);
    first_channel.convertTo(src_f, CV_32FC1, 1.0 / 255.0);

    cv::Mat srcB(rows, cols, CV_32FC1);
    cv::Mat srcD(rows, cols, CV_32FC1);

    // 背景层细节层分离
    cv::boxFilter(src_f, srcB, -1, cv::Size(2 * size + 1, 2 * size + 1));

    cv::Mat log_src(rows, cols, CV_32FC1);
    cv::Mat log_srcB(rows, cols, CV_32FC1);
    cv::Mat log_srcD(rows, cols, CV_32FC1);

    cv::log(src_f, log_src);
    cv::log(srcB, log_srcB);
    log_srcD = log_src - log_srcB;
    cv::exp(log_srcD, srcD);

    // 细节增强
    cv::Mat srcD_en(rows, cols, CV_32FC1);
    cv::pow(srcD, en, srcD_en);

    // 降噪
    cv::Mat srcD_deno(rows, cols, CV_32FC1);
    if (filtertype == 1)
    {
        cv::medianBlur(srcD_en, srcD_deno, filtersize);
    }
    else if (filtertype == 2)
    {
        cv::GaussianBlur(srcD_en, srcD_deno, cv::Size(filtersize, filtersize), 0);
    }

    // 合并
    cv::Mat log_srcD_en(rows, cols, CV_32FC1);
    cv::Mat log_dst(rows, cols, CV_32FC1);
    cv::Mat dst_f(rows, cols, CV_32FC1);

    cv::log(srcD_deno, log_srcD_en);
    log_dst = log_srcB + log_srcD_en;
    cv::exp(log_dst, dst_f);

    dst_f.convertTo(first_channel, CV_8UC1, 255);

    // 将处理后的通道合并回图像
    channels[0] = first_channel;
    cv::merge(channels, dst);
}

U8 ytxi_process(YUV* yuv, IMG_CONTEXT context, G_CONFIG cfg)
{
	cv::Mat src(context.height, context.width, CV_8UC3);
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			src.at<cv::Vec3b>(y, x)[0] = yuv[index].y;
			src.at<cv::Vec3b>(y, x)[1] = yuv[index].u;
			src.at<cv::Vec3b>(y, x)[2] = yuv[index].v;
		}
	}
   
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

	//细节层半径
	int size = 4;
	//细节增强强度>1
	float str = 2;
	//滤波方式：1=中值滤波，2=高斯滤波
	int filtertype = 1;
	//滤波器大小
	int filtersize = 3;

	int t0 = clock();
	YTXI(src, dst, context.height, context.width, size, str, filtertype, filtersize);

	int t1 = clock();
	cout << "TXI time = " << (t1 - t0) / 1000.0 << "s" << endl;

	// 将 dst 转回 YUV 格式
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			yuv[index].y = dst.at<cv::Vec3b>(y, x)[0];
		}
	}

    RGB *rgb_data = y2r_process(yuv, context, cfg);

    save_img_with_timestamp(rgb_data, &context, "_ytxi");
    LOG("done.");
    return OK;
}

void TXI_nv12(cv::Mat src, cv::Mat dst, int rows, int cols, int size, float en, int filtertype, int filtersize) {
    // 提取前 2/3 的数据转换为新的 cv::Mat 对象
    cv::Mat y_mat(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            y_mat.at<uchar>(i, j) = src.at<uchar>(i * cols + j);
        }
    }

    save_mat(y_mat, "0_src.jpg");
    cv::Mat src_f(rows, cols, CV_32FC1);
    y_mat.convertTo(src_f, CV_32FC1, 1.0 / 255.0);
    //save_mat(src_f, "0_src_f.jpg");

    cv::Mat srcB(rows, cols, CV_32FC1);
    cv::Mat srcD(rows, cols, CV_32FC1);

    // 背景层细节层分离
    cv::boxFilter(src_f, srcB, -1, cv::Size(2 * size + 1, 2 * size + 1));
    save_mat(srcB, "1_bg.jpg");

    cv::Mat log_src(rows, cols, CV_32FC1);
    cv::Mat log_srcB(rows, cols, CV_32FC1);
    cv::Mat log_srcD(rows, cols, CV_32FC1);

    cv::log(src_f, log_src);
    cv::log(srcB, log_srcB);
    log_srcD = log_src - log_srcB;
    cv::exp(log_srcD, srcD);
    save_mat(srcD, "2_detail.jpg");

    // 细节增强
    cv::Mat srcD_en(rows, cols, CV_32FC1);
    cv::pow(srcD, en, srcD_en);
    save_mat(srcD_en, "3_enhance.jpg");

    // 降噪
    cv::Mat srcD_deno(rows, cols, CV_32FC1);
    if (filtertype == 1) {
        cv::medianBlur(srcD_en, srcD_deno, filtersize);
    }
    else if (filtertype == 2) {
        cv::GaussianBlur(srcD_en, srcD_deno, cv::Size(filtersize, filtersize), 0);
    }
    save_mat(srcD_deno, "4_deno.jpg");

    // 合并
    cv::Mat log_srcD_en(rows, cols, CV_32FC1);
    cv::Mat log_dst(rows, cols, CV_32FC1);
    cv::Mat dst_f(rows, cols, CV_32FC1);

    cv::log(srcD_deno, log_srcD_en);
    log_dst = log_srcB + log_srcD_en;
    cv::exp(log_dst, dst_f);

    dst_f.convertTo(y_mat, CV_8UC1, 255);
    save_mat(y_mat, "5_dst.jpg");

    // 将处理后的 y_mat 写回 dst
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst.at<uchar>(i * cols + j) = y_mat.at<uchar>(i, j);
        }
    }

    //UV数据直接沿用src
    for (int i = rows * cols; i < rows * cols * 1.5; ++i) 
    {
        dst.at<uchar>(i) = src.at<uchar>(i);
    }
}


U8 txi_nv12_process(YUV* yuv, IMG_CONTEXT context, G_CONFIG cfg)
{
    // 计算 YUV 数据大小
    int width = context.width;
    int height = context.height;

    // 创建 Y、U、V 平面
    cv::Mat yPlane(height, width, CV_8UC1);
    cv::Mat uvPlane(height / 2, width / 2, CV_8UC2);

    // 分离 Y、U、V 数据
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            yPlane.at<U8>(y, x) = yuv[index].y;
        }
    }

    int uvIndex = 0;
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width / 2; ++x) {
            int index = (y * 2 * width + x * 2);
            uvPlane.at<cv::Vec2b>(y, x)[0] = yuv[index].u;
            uvPlane.at<cv::Vec2b>(y, x)[1] = yuv[index].v;
            uvIndex++;
        }
    }

    // 合并 Y 和 UV 平面为 NV12 格式
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
    yPlane.copyTo(nv12(cv::Rect(0, 0, width, height)));
    uvPlane.reshape(1, height / 2).copyTo(nv12(cv::Rect(0, height, width, height / 2)));

    
    //细节层半径
    int size = 4;
    //细节增强强度>1
    float str = 2;
    //滤波方式：1=中值滤波，2=高斯滤波
    int filtertype = 1;
    //滤波器大小
    int filtersize = 3;

    int t0 = clock();

    cv::Mat dst = cv::Mat::zeros(nv12.size(), nv12.type());
    TXI_nv12(nv12, dst, context.height, context.width, size, str, filtertype, filtersize);

    int t1 = clock();
    cout << "TXI time = " << (t1 - t0) / 1000.0 << "s" << endl;


    // 将 NV12 格式数据转回 YUV 格式
    cv::Mat yuvMat(height, width, CV_8UC3);
    cv::cvtColor(dst, yuvMat, cv::COLOR_YUV2BGR_NV12);

    // 将 cv::Mat 数据复制回 YUV 结构体
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            cv::Vec3b pixel = yuvMat.at<cv::Vec3b>(y, x);
            S32 r = pixel[2];
            S32 g = pixel[1];
            S32 b = pixel[0];

            // 转换为 YUV，使用整数计算
            S32 y_val = ((S32)77 * r + 150 * g + 29 * b) >> 8;  // Y = (0.299*256 * R + 0.587*256 * G + 0.114*256 * B) / 256
            S32 u_val = (((S32)-43 * r - 85 * g + 128 * b) >> 8) + 128; // U = (−0.147*256 * R − 0.289*256 * G + 0.436*256 * B) / 256 + 128
            S32 v_val = (((S32)128 * r - 107 * g - 21 * b) >> 8) + 128; // V = (0.615*256 * R − 0.515*256 * G − 0.100*256 * B) / 256 + 128


            yuv[index].y = y_val;
            yuv[index].u = u_val;
            yuv[index].v = v_val;
        }
    }

    RGB* rgb_data = y2r_process(yuv, context, cfg);

    save_img_with_timestamp(rgb_data, &context, "_txinv12");
    LOG("done.");
    return OK;

}
