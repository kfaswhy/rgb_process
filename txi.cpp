#include "txi.h"

using namespace std;
using namespace cv;
U8 txi_process(RGB* rgb, IMG_CONTEXT context, G_CONFIG cfg) 
{
	// 将 RGB* rgb 转为 Mat src，CV_8UC3 格式
	cv::Mat src(context.height, context.width, CV_8UC3);
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			src.at<cv::Vec3b>(y, x)[0] = rgb[index].b;
			src.at<cv::Vec3b>(y, x)[1] = rgb[index].g;
			src.at<cv::Vec3b>(y, x)[2] = rgb[index].r;
		}
	}
	
	// 新建空 Mat dst，和 src 同属性
	cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

	//细节层半径
	int size = 4;
	//细节增强强度>1
	float en = 2;
	//滤波方式：1=中值滤波，2=高斯滤波
	int filtertype = 1; 
	//滤波器大小
	int filtersize = 3;

	int t0 = clock();
	TXI(src, dst, context.height, context.width, size, en, filtertype, filtersize);

	int t1 = clock();
	cout << "TXI time = " << (t1 - t0) / 1000.0 << "s" << endl;
	// 将 dst 转回 RGB 格式并保存到 RGB* rgb 中
	for (int y = 0; y < context.height; ++y) {
		for (int x = 0; x < context.width; ++x) {
			int index = y * context.width + x;
			rgb[index].b = dst.at<cv::Vec3b>(y, x)[0];
			rgb[index].g = dst.at<cv::Vec3b>(y, x)[1];
			rgb[index].r = dst.at<cv::Vec3b>(y, x)[2];
		}
	}

	return OK;
}

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

void printMatInfo(const cv::Mat& mat) {
	// 打印数值类型
	int type = mat.type();
	std::string typeStr;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch (depth) {
	case CV_8U:
		typeStr = "8U";
		break;
	case CV_8S:
		typeStr = "8S";
		break;
	case CV_16U:
		typeStr = "16U";
		break;
	case CV_16S:
		typeStr = "16S";
		break;
	case CV_32S:
		typeStr = "32S";
		break;
	case CV_32F:
		typeStr = "32F";
		break;
	case CV_64F:
		typeStr = "64F";
		break;
	default:
		typeStr = "unknown";
		break;
	}
	typeStr += "C";
	typeStr += (chans + '0');
	std::cout << "Mat type: " << typeStr << std::endl;

	// 打印最大值和最小值
	double minVal, maxVal;
	cv::minMaxLoc(mat, &minVal, &maxVal);
	std::cout << "Min value: " << minVal << std::endl;
	std::cout << "Max value: " << maxVal << std::endl;
}

void TXI(cv::Mat src, cv::Mat dst, int rows, int cols, int size, float en, int filtertype, int filtersize)
{
	save_mat(src, "0_src.jpg");
	cv::Mat src_f(rows, cols, CV_32FC3);
	
	
	src.convertTo(src_f, CV_32FC3, 1.0 / 255.0);
	save_mat(src_f, "0_src_f.jpg");
	
    
	cv::Mat srcB(rows, cols, CV_32FC3);
	cv::Mat srcD(rows, cols, CV_32FC3);

	//背景层细节层分离
	boxFilter(src_f, srcB, -1, cv::Size(2 * size + 1, 2 * size + 1));


	save_mat(srcB, "1_bg.jpg");
	cv::Mat log_src(rows, cols, CV_32FC3);
	cv::Mat log_srcB(rows, cols, CV_32FC3);
	cv::Mat log_srcD(rows, cols, CV_32FC3);

	log(src_f, log_src);
	log(srcB, log_srcB);
	log_srcD = log_src - log_srcB;
	exp(log_srcD, srcD);
	save_mat(srcD, "2_detail.jpg");


	//细节增强
	cv::Mat srcD_en(rows, cols, CV_32FC3);
	pow(srcD, en, srcD_en);
	save_mat(srcD_en, "3_enhance.jpg");

	//降噪
	cv::Mat srcD_deno(rows, cols, CV_32FC3);
	if (filtertype = 1)
	{
		medianBlur(srcD_en, srcD_deno, filtersize);
	}
	else if (filtertype = 2)
	{
		GaussianBlur(srcD_en, srcD_deno, cv::Size(filtersize, filtersize), 0);
	}

	save_mat(srcD_deno, "4_deno.jpg");

	//合并
	cv::Mat log_srcD_en(rows, cols, CV_32FC3);
	cv::Mat log_dst(rows, cols, CV_32FC3);
	cv::Mat dst_f(rows, cols, CV_32FC3);

	log(srcD_deno, log_srcD_en);
	

	log_dst = log_srcB + log_srcD_en;
	exp(log_dst, dst_f);

	dst_f.convertTo(dst, CV_8UC3, 255);
	save_mat(dst, "5_dst.jpg");
}





////导向滤波图像增强
//extern "C" __declspec(dllexport) void DetailEnhance_cpp(const uchar* src, uchar* dst, 
//	int width, int height, int cn, int radius, float eps, float lamda/*, float *t*/)
//{
//	//变量定义
//	int x, y, pix_size, dwidth, dheight, srcStep, IStep, dstStep, * x_ofs, len;
//	float* I, * tmp, * tmp1, * tmp2, * upA, * upB;
//
//#pragma region 下采样
//	double ifx, ify;
//	pix_size = cn;
//	eps = eps * 255 * 255;
//	ifx = SCALE;
//	ify = SCALE;
//	dwidth = width >> 2;
//	dheight = height >> 2;
//	I = (float*)AllocMemory(dheight * dwidth * cn * sizeof(float), false);
//	tmp = (float*)AllocMemory(dheight * dwidth * cn * sizeof(float), false);
//	tmp1 = (float*)AllocMemory(dheight * dwidth * cn * sizeof(float), false);
//	tmp2 = (float*)AllocMemory(dheight * dwidth * cn * sizeof(float), false);
//	upA = (float*)AllocMemory(height * width * cn * sizeof(float), false);
//	upB = (float*)AllocMemory(height * width * cn * sizeof(float), false);
//	if (NULL == I) return;
//	x_ofs = (int*)malloc(dwidth * sizeof(int));
//	if (NULL == x_ofs) return;
//	dstStep = srcStep = width * cn;
//	IStep = dwidth * cn;
//	len = width * height * cn;
//	for (x = 0; x < dwidth; ++x) {
//		int sx = std::floor(x * ifx);
//		x_ofs[x] = min(sx, width - 1) * pix_size;
//	}
//#pragma omp parallel for
//	for (y = 0; y < dheight; ++y) {
//		float* D = (float*)(I + IStep * y);
//		int sy = min(std::floor(y * ify), (double)height - 1);
//		const uchar* S = src + srcStep * sy;
//		for (x = 0; x < dwidth; ++x, D += cn) {
//			const uchar* _tS = S + x_ofs[x];
//			D[0] = float(_tS[0]); D[1] = float(_tS[1]); D[2] = float(_tS[2]);
//		}
//	}
//#pragma endregion
//
//#pragma region 导向滤波
//	BoxFilter(I, tmp, dwidth, dheight, cn, radius);     // meanI = fmean(I, r); tmp = meanI;
//
//	Mul(I, I, tmp1, dwidth, dheight, cn);               // tmp1 = I .* I;
//
//	BoxFilter(tmp1, tmp2, dwidth, dheight, cn, radius);	//corrI = fmean(I.*I, r); tmp2 = corrI;
//	Mul(tmp, tmp, tmp1, dwidth, dheight, cn);           // tmp1 = meanI .* meanI;
//
//	Sub(tmp2, tmp1, tmp2, dwidth, dheight, cn);         // varI = corrI - meanI .* meanI; tmp2 = varI;
//	Div(tmp2, tmp2, tmp2, eps, dwidth, dheight, cn);    // a = covIp ./ (varI + eps) tmp2 = a; covIp = varI;
//
//	Mul(tmp2, tmp, tmp1, dwidth, dheight, cn);          // tmp1 = a .* meanI;
//	Sub(tmp, tmp1, tmp1, dwidth, dheight, cn);          // b = meanI - a .* meanI; tmp1 = b;
//
//	BoxFilter(tmp2, tmp, dwidth, dheight, cn, radius);  // meana = fmean(a, r); tmp = meana;
//	BoxFilter(tmp1, tmp2, dwidth, dheight, cn, radius); // meanb = fmean(b, r); tmp2 = meanb;
//#pragma endregion
//
//	//上采样
//	upResizeTwo(tmp, tmp2, upA, upB, dwidth, dheight, width, height, cn);
//
//#pragma region 复制增强结果
//	//采用多线程计算增强结果并复制数据
//	//HANDLE hThread[Thread_Num];
//	//struct threadPack TPack;
//	//TPack.lamda = lamda;
//	//int i = 0;
//	//int Step = srcStep * height / Thread_Num;
//	//TPack.len = Step;
//	//TPack.input = src; TPack.output = dst; TPack.upA = upA; TPack.upB = upB;
//	//hCopyThread = CreateEvent(NULL, FALSE, FALSE, NULL);
//	//while (i < Thread_Num) {
//	//	TPack.idx = i;
//	//	hThread[i] = (HANDLE)_beginthreadex(NULL, 0, copyThread, &TPack, 0, NULL);
//	//	WaitForSingleObject(hCopyThread, INFINITE);   //等待事件被触发  
//	//	++i;
//	//}
//	//WaitForMultipleObjects(Thread_Num, hThread, true, INFINITE);
//	//CloseHandle(hCopyThread);
//
//	//直接计算增强结果并复制数据
//#pragma omp parallel for
//	for (x = 0; x < len; x++) {
//		dst[x] = clampToByte(src[x] * lamda + (src[x] * upA[x] + upB[x]) * (1 - lamda));
//	}
//#pragma endregion
//
//	//释放内存
//	FreeMemory(upA);
//	FreeMemory(upB);
//	FreeMemory(I);
//	FreeMemory(tmp);
//	FreeMemory(tmp1);
//	FreeMemory(tmp2);
//	free(x_ofs);
//}
//
//_declspec(dllexport) void TXI(uchar* pSrcbuf, uchar* pDstbuf, int rows, int cols, int size, float en, int filtertype, int filtersize)
//{
//	Mat src(rows, cols, CV_8UC3, pSrcbuf);
//	Mat src_f(rows, cols, CV_32FC3);
//	Mat dst(rows, cols, CV_8UC3);
//
//	src.convertTo(src_f, CV_32FC3, 1.0 / 255.0);
//	Mat srcB(rows, cols, CV_32FC3);
//	Mat srcD(rows, cols, CV_32FC3);
//
//	//背景层细节层分离
//	boxFilter(src_f, srcB, -1, Size(2 * size + 1, 2 * size + 1));
//	Mat log_src(rows, cols, CV_32FC3);
//	Mat log_srcB(rows, cols, CV_32FC3);
//	Mat log_srcD(rows, cols, CV_32FC3);
//
//	log(src_f, log_src);
//	log(srcB, log_srcB);
//	log_srcD = log_src - log_srcB;
//	exp(log_srcD, srcD);
//
//	//细节增强
//	Mat srcD_en(rows, cols, CV_32FC3);
//	pow(srcD, en, srcD_en);
//
//	//降噪
//	Mat srcD_deno(rows, cols, CV_32FC3);
//	if (filtertype = 1)
//	{
//		medianBlur(srcD_en, srcD_deno, filtersize);
//	}
//	else if (filtertype = 2)
//	{
//		GaussianBlur(srcD_en, srcD_deno, Size(filtersize, filtersize), 0);
//	}
//
//	//合并
//	Mat log_srcD_en(rows, cols, CV_32FC3);
//	Mat log_dst(rows, cols, CV_32FC3);
//	Mat dst_f(rows, cols, CV_32FC3);
//
//	log(srcD_deno, log_srcD_en);
//	log_dst = log_srcB + log_srcD_en;
//	exp(log_dst, dst_f);
//
//	dst_f.convertTo(dst, CV_8UC3, 255);
//
//	memcpy(pDstbuf, dst.data, sizeof(uchar) * rows * cols);
//}
