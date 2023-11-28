//#include<opencv2/opencv.hpp>
//#include<iostream>
//#include<vector>
//#include<iomanip>
//#include<cmath>
//#include<string>
//
//using namespace std;
//using namespace cv;
//double pi = 3.1415926;
//
//// 累计概率直方图
//void SumRatio(Mat image,float accumulate_ratio[256])
//{	
//	int rows = image.rows;
//	int cols = image.cols;
//	double numpixels = image.total();
//	/*for (int r = 0; r < rows; r++)
//	{
//		for (int c = 0; c < cols; c++)
//		{
//			int pixval = image.at<unsigned short>(r, c);
//			pixval = pixval / 256;
//			image.at<unsigned short>(r, c) = int(pixval);
//		}
//	}*/
//	int hist[256] = { 0 };  // 灰度直方图
//	int pixval;
//	for (int i = 0; i < rows; i++)
//	{
//		for (int j = 0; j < cols; j++)
//		{
//			pixval = image.at<unsigned char>(i, j);
//			hist[pixval] += 1;
//		}
//	}
//	
//	// 计算每个灰度值出现的概率
//	float ratio[256] = { 0 };
//	for (int k = 0; k < 256; k++)
//	{
//		ratio[k] = hist[k] / numpixels;
//	}
//
//	// 计算累计概率直方图
//	//float accumulate_ratio[256] = { 0 };
//	float sum = 0;
//	for (int m = 0; m < 256; m++)
//	{
//		sum = sum + ratio[m];
//		accumulate_ratio[m] = sum;
//	}
//}
//
//// 计算累计直方图的差异，并生成灰度值映射关系
//void map_color(Mat src, Mat refer,float PixelMaps[256])
//{
//	//累计概率直方图
//	float ratio_src[256] = { 0 };
//	float ratio_refer[256] = { 0 };
//	SumRatio(src, ratio_src);
//	/*for (int k = 0; k < 256; k++)
//	{
//		cout << ratio_src[k] << endl;
//	}*/
//	SumRatio(refer, ratio_refer);
//	//遍历原图每一个灰度的累计概率
//	//float color_map[256] = { 0 };
//	for (int i = 0; i < 256; i++) 
//	{
//		float min = 100;
//		int referColor = 0;
//		float diff;
//		for (int n = 0; n < 256; n++)
//		{	
//			float diff = abs(ratio_src[i] - ratio_refer[n]);
//			//cout << "diff:" << diff << " min:" << min << " 此时n= " << n << endl;
//			if (diff < min) 
//			{
//				min = diff;
//				referColor = n;
//			}
//		}
//		PixelMaps[i] = referColor;
//	}
//}
//
//Mat ColorMatch(Mat src, Mat reference)
//{
//	/*
//	直方图匹配： 让一张图参考另一张图， 让他们的灰度保持一致
//	步骤：
//	计算原图累计直方图
//	计算参考图的累计直方图
//	计算两个累计直方图的差异
//	生成原图和参考图之间的灰度映射
//	*/
//	float PixelMaps[256] = { 0 };
//	map_color(src, reference, PixelMaps);
//	/*for (int k = 0; k < 256; k++)
//	{
//		cout << PixelMaps[k] << endl;
//	}*/
//	Mat result = src;
//	int height = result.rows;
//	int width = result.cols;
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			int pixval = src.at<unsigned char>(i, j);
//			int refercolor = PixelMaps[pixval];
//			result.at<unsigned char>(i, j) = refercolor;
//		}
//	}
//	return result;
//}
//
//
//
//// 取图像的中值
//float median(Mat input)
//{
//	Mat rmat = input.reshape(1, 1);
//	Mat s;
//	cv::sort(rmat, s, CV_SORT_EVERY_ROW);
//	float MidValue = s.at<unsigned char>(0, (s.cols - 1) / 2);
//	return MidValue;
//}
//
//
//void gauss(int size, double sigma, double f[])
//{
//	int x = size / 2;
//	double sum = 0.0;
//	// 公式
//	for (int i = 0; i < size; i++)
//		for (int j = 0; j < size; j++)
//		{
//			f[i*size + j] = 1 / (2 * pi*sigma*sigma)*exp((-1) * ((i - x)*(i - x) + (j - x)*(j - x)) / (2 * sigma*sigma));
//			sum = sum + f[i*size + j];   //权重总和
//		}
//	//归一化
//	for (int i = 0; i < size; i++)
//	{
//		for (int j = 0; j < size; j++)
//		{
//			f[i*size + j] = f[i*size + j] / sum;
//			cout << f[i*size + j] << "    ";
//		}
//		cout << endl;
//	}
//}
//
//Mat conv2D(Mat src,int size, double sigma)
//{
//	double filter[1] = {0};
//	gauss(size, sigma, filter);
//	int borderType = BORDER_DEFAULT;
//	Point anchor = Point(-1, -1);
//	Mat kernel = Mat_<double>(size, size, filter);
//	//Mat kernel = (Mat_<float>(2, 2) << 0.07511361, 0.1238414, 0.07511361, 0.1238414, 0.20417996, 0.1238414, 0.07511361, 0.1238414, 0.07511361);
//	Mat kernelFilp;
//	Mat dst;
//	Mat res;
//	flip(kernel, kernelFilp, -1);
//	filter2D(src, dst, -1, kernel, anchor, 0.0, borderType);
//	//res = dst + 100;
//	return dst;
//}
//
//// 平滑函数
////def smooth2nd(x, M) : ##x 为一维数组
////K = round(M / 2 - 0.1) ##M应为奇数，如果是偶数，则取大1的奇数
////lenX = len(x)
////if lenX < 2 * K + 1 :
////	print('数据长度小于平滑点数')
////else :
////	y = np.zeros(lenX)
////	for NN in range(0, lenX, 1) :
////		startInd = max([0, NN - K])
////		endInd = min(NN + K + 1, lenX)
////		y[NN] = np.mean(x[startInd:endInd])
////##    y[0]=x[0]       #首部保持一致
////##    y[-1]=x[-1]     #尾部也保持一致
////		return(y)
//
//Mat smoothdata(Mat input, int size)
//{
//	int rows = input.rows;
//	int cols = input.cols;
//	Mat result = Mat::zeros(Size(cols, rows), CV_8U);
//	int k = int(size / 2 - 1);
//	for (int c = 0; c < cols; c++)
//	{
//		int start = max({ 0, c - k });
//		int end = min({ c + k + 1, cols });
//		//cout << "start: " << start << "end: " << end << endl;
//		Mat sildewindow = input(Range(0, 1), Range(start, end));
//		Scalar n;
//		Mat m;
//		for (int i = 0; i < sildewindow.cols; i++)
//		{
//			int mPixval = sildewindow.at<unsigned char>(0, i);
//			if (mPixval >  median(sildewindow)*1.5)
//			{
//				sildewindow.at<unsigned char>(0, i) = median(sildewindow);
//			}
//		}
//		GaussianBlur(sildewindow,m,Size(1,1),0,0);
//		
//		//GaussianBlur(m, m, Size(1, 1), 0, 0);
//		n = mean(m);
//		result.at<unsigned char>(0, c) = n[0];
//	}
//	return result;
//}
//
//int main()
//{
//	int a = 257;
//	int b = 288;
//	Mat src = imread("./images/png/DRImage_Beam02_2.png", CV_LOAD_IMAGE_UNCHANGED);
//	int row = src.rows;
//	int col = src.cols;
//	for (int r = 0; r < row; r++)
//	{
//		for (int c = 0; c < col; c++)
//		{
//			int pixval = src.at<unsigned short>(r, c);
//			pixval = pixval / 256;
//			src.at<unsigned short>(r, c) = int(pixval);
//		}
//	}
//	src.convertTo(src, CV_8U);
//
//
//	Mat src1 = src;
//	int rows = src1.rows;
//	int cols = src1.cols;
//	for (int i = 0; i < rows; i++) 
//	{
//		Mat e2 = Mat::zeros(Size(60, 1), CV_8U);
//		Mat roi1 = src(Range(i, i + 1),Range(a - 30, a));  // image(Range(行始，行终),Range(列始，列终))
//		Mat roi2 = src(Range(i, i + 1),Range(b + 1, b + 31));
//		// 拼接两个roi区域
//		roi1.copyTo(e2(Range(0, 1),Range(0, 30)));
//		roi2.copyTo(e2(Range(0, 1),Range(30, 60)));
//		// 坏线的区域
//		Mat bad = src(Range(i, i+1),Range(a, b+1));
//		// 直方图匹配
//		Mat I1;
//		I1 = ColorMatch(bad, e2);
//		/*for (int j = 0; j < 1; j++)
//		{
//			for (int k = 0; k < 32; k++)
//			{
//				int pixvel = I1.at<unsigned short>(j, k);
//				pixvel = pixvel * 256;
//				I1.at<unsigned short>(j, k) = pixvel;
//			}
//		}*/
//		I1.copyTo(src1(Range(i, i+1),Range(a, b+1)));
//	}
//	
//	//namedWindow("直方图匹配结果", CV_WINDOW_NORMAL);
//	//imshow("直方图匹配结果", src1);
//	//waitKey(0);
//
//	Mat src2 = src1;
//	imwrite("test.png", src2);
//	int offset = 30;
//	int offset1 = 15;
//	int nnn = b + offset - a + offset + 1 ;
//	for (int j = 0; j < rows; j++)
//	{
//		Mat aa = src2(Range(j, j + 1),Range(a - offset, b + offset + 1));
//		Mat aaz = src2(Range(j, j + 1),Range(a - offset1, a));
//		Mat aay = src2(Range(j, j + 1),Range(b + 1, b + offset1 +1));
//		for (int k = 1; k < nnn; k++)
//		{
//			//cout << MidValue << endl;
//			if ((abs(aa.at<unsigned char>(0, k) - aa.at<unsigned char>(0, k - 1)) / aa.at<unsigned char>(0, k - 1)) > 0.8)
//			{
//				aa.at<unsigned char>(0, k) = median(aa);
//			}
//			else if (aa.at<unsigned char>(0, k) < (median(aa) / 2))
//			{
//				aa.at<unsigned char>(0, k) = median(aa);
//			}
//			else
//			{
//				aa.at<unsigned char>(0, k) = aa.at<unsigned char>(0, k);
//			}
//		}
//		// 高斯平滑
//		Mat after_gauss;
//		Mat before_gauss = src2(Range(j, j + 1), Range(a - offset, b + offset + 1));
//		after_gauss = smoothdata(before_gauss, 10);
//		after_gauss.copyTo(src2(Range(j, j + 1), Range(a - offset, b + offset + 1)));
//	}
//	/*namedWindow("高斯平滑后的结果", CV_WINDOW_NORMAL);
//	imshow("高斯平滑后的结果", src2);
//	waitKey(0);*/
//	
//	
//	 // 灰度直方图均衡化
//	Mat result;
//	/*for (int m = 0; m < rows; m++)
//	{
//		for (int n = 0; n < cols; n++)
//		{
//			int pxv = src2.at<unsigned short>(m, n);
//			pxv = pxv / 256;
//			src2.at<unsigned short>(m, n) = int(pxv);
//		}
//	}
//	src2.convertTo(src2, CV_8U);*/
//	equalizeHist(src2, result);
//	namedWindow("直方图均衡化后的结果", CV_WINDOW_NORMAL);
//	imshow("直方图均衡化后的结果", result);
//	waitKey(0);
//	destroyAllWindows;
//
//	return 0;
//}
//

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<iomanip>
#include<cmath>
#include<string>
#include<ctime>
#include<thread>

using namespace std;
using namespace cv;
double pi = 3.1415926;

// 累计概率直方图
void SumRatio(Mat image, float accumulate_ratio[128])
{
	int rows = image.rows;
	int cols = image.cols;
	double numpixels = image.total();
	/*for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int pixval = image.at<unsigned short>(r, c);
			pixval = pixval / 256;
			image.at<unsigned short>(r, c) = int(pixval);
		}
	}*/
	int hist[128] = { 0 };  // 灰度直方图
	int pixval;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			pixval = image.at<unsigned short>(i, j);
			hist[pixval] += 1;
		}
	}

	// 计算每个灰度值出现的概率
	float ratio[128] = { 0 };
	for (int k = 0; k < 128; k++)
	{
		ratio[k] = hist[k] / numpixels;
	}

	// 计算累计概率直方图
	//float accumulate_ratio[256] = { 0 };
	float sum = 0;
	for (int m = 0; m < 128; m++)
	{
		sum = sum + ratio[m];
		accumulate_ratio[m] = sum;
	}
}

// 计算累计直方图的差异，并生成灰度值映射关系
void map_color(Mat src, Mat refer, float PixelMaps[128])
{
	//累计概率直方图
	float ratio_src[128] = { 0 };
	float ratio_refer[128] = { 0 };
	SumRatio(src, ratio_src);
	/*for (int k = 0; k < 256; k++)
	{
		cout << ratio_src[k] << endl;
	}*/
	SumRatio(refer, ratio_refer);
	//遍历原图每一个灰度的累计概率
	//float color_map[256] = { 0 };
	for (int i = 0; i < 128; i++)
	{
		float min = 100;
		int referColor = 0;
		float diff;
		for (int n = 0; n < 128; n++)
		{
			float diff = abs(ratio_src[i] - ratio_refer[n]);
			//cout << "diff:" << diff << " min:" << min << " 此时n= " << n << endl;
			if (diff < min)
			{
				min = diff;
				referColor = n;
			}
		}
		PixelMaps[i] = referColor;
	}
}



Mat ColorMatch(Mat src, Mat reference)
{
	/*
	直方图匹配： 让一张图参考另一张图， 让他们的灰度保持一致
	步骤：
	计算原图累计直方图
	计算参考图的累计直方图
	计算两个累计直方图的差异
	生成原图和参考图之间的灰度映射
	*/
	float PixelMaps[128] = { 0 };
	clock_t start, finish;
	//start = clock();
	map_color(src, reference, PixelMaps);
	//finish = clock();
	//double time_sub = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "灰度值映射所花时间： " << time_sub << endl;
	/*for (int k = 0; k < 256; k++)
	{
		cout << PixelMaps[k] << endl;
	}*/
	Mat result = src;
	int height = result.rows;
	int width = result.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int pixval = src.at<unsigned short>(i, j);
			int refercolor = PixelMaps[pixval];
			result.at<unsigned short>(i, j) = refercolor;
		}
	}
	return result;
}


// 取图像的中值
float median(Mat input)
{
	Mat rmat = input.reshape(1, 1);
	Mat s;
	cv::sort(rmat, s, CV_SORT_EVERY_ROW);
	float MidValue = s.at<unsigned short>(0, (s.cols - 1) / 2);
	return MidValue;
}

Mat smoothdata(Mat input, int size)
{
	int rows = input.rows;
	int cols = input.cols;
	Mat result = Mat::zeros(Size(cols, rows), CV_16U);
	int k = int(size / 2 - 1);
	for (int c = 0; c < cols; c++)
	{
		int start = max({ 0, c - k });
		int end = min({ c + k + 1, cols });
		//cout << "start: " << start << "end: " << end << endl;
		Mat sildewindow = input(Range(0, 1), Range(start, end));
		Scalar n;
		Mat m;
		float MidVal = median(sildewindow);
		for (int i = 0; i < sildewindow.cols; i++)
		{
			int mPixval = sildewindow.at<unsigned short>(0, i);
			if (mPixval > MidVal*1.2)  // 1.5
			{
				sildewindow.at<unsigned short>(0, i) = MidVal;
			}
		}
		GaussianBlur(sildewindow, m, Size(1, 1), 0, 0);

		//GaussianBlur(m, m, Size(1, 1), 0, 0);
		n = mean(m);
		result.at<unsigned short>(0, c) = n[0];
	}
	return result;
}


Mat bit16Tobit8(Mat image)
{
	int rows = image.rows;
	int cols = image.cols;
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int PixVal = image.at<unsigned short>(r, c);
			PixVal = PixVal / 512;
			image.at<unsigned short>(r, c) = PixVal;
		}
	}
	return image;
}


void renovation(Mat src, int *start, int *end, int line_nums,int offset)
{
	//Mat src1 = src;

	clock_t start1, finish1, start2, finish2;
	// 对每一条坏线进行处理
	start1 = clock();
	for (int aL = 0; aL < line_nums; aL++)
	{
		int rows = src.rows;
		int cols = src.cols;
		//// 新建一张与原图尺寸一致的图
		//Mat temp = Mat::zeros(Size(cols, rows), CV_16U);
		//// 新建一张与坏线区域及其周围区域大小一致的图
		//Mat temp1 = Mat::zeros(Size(end[aL] + offset - start[aL] + offset + 1, rows), CV_16U);
		for (int i = 0; i < rows; i++)
		{
			Mat e2 = Mat::zeros(Size(2 * offset, 1), CV_16U);
			Mat roi1 = src(Range(i, i + 1), Range(start[aL] - offset, start[aL]));  // image(Range(行始，行终),Range(列始，列终))
			Mat roi2 = src(Range(i, i + 1), Range(end[aL] + 1, end[aL] + offset + 1));
			// 拼接两个roi区域
			roi1.copyTo(e2(Range(0, 1), Range(0, offset)));
			roi2.copyTo(e2(Range(0, 1), Range(offset, 2 * offset)));
			// 坏线的区域
			Mat bad = src(Range(i, i + 1), Range(start[aL], end[aL] + 1));
			e2 = bit16Tobit8(e2);
			bad = bit16Tobit8(bad);
			
			// 直方图匹配
			Mat I1;
			I1 = ColorMatch(bad, e2);
			for (int j = 0; j < 1; j++)
			{
				for (int k = 0; k < end[aL]-start[aL]+1; k++)
				{
					int pixvel = I1.at<unsigned short>(j, k);
					pixvel = pixvel * 512;
					I1.at<unsigned short>(j, k) = pixvel;
				}
			}
			I1.copyTo(src(Range(i, i + 1), Range(start[aL], end[aL] + 1)));
		}
		finish1 = clock();
		double duration1 = (double)(finish1 - start1) / CLOCKS_PER_SEC;
		cout << "直方图匹配花费时间：" << duration1 << endl;
		imwrite("直方图匹配的结果.png", src);
		namedWindow("直方图匹配结果", CV_WINDOW_NORMAL);
		imshow("直方图匹配结果", src);
		waitKey(0);

		//int offset = 30;
		int offset1 = 15;
		int nnn = end[aL] + offset - start[aL] + offset + 1;


		start2 = clock();
		for (int j = 0; j < rows; j++)
		{
			Mat aa = src(Range(j, j + 1), Range(start[aL] - offset, end[aL] + offset + 1));
			Mat aaz = src(Range(j, j + 1), Range(start[aL] - offset1, start[aL]));
			Mat aay = src(Range(j, j + 1), Range(end[aL] + 1, end[aL] + offset1 + 1));
			for (int k = 1; k < nnn; k++)
			{
				// cout << MidValue << endl;
				float midval = median(aa);
				int temp_pixelk = aa.at<unsigned short>(0, k);
				int temp_pixelk1 = aa.at<unsigned short>(0, k - 1);
				if (temp_pixelk1 == 0 )
				{
					aa.at<unsigned short>(0, k) = temp_pixelk;
				}
				else if ((abs(temp_pixelk - temp_pixelk1) / temp_pixelk1) > 0.8)
				{
					aa.at<unsigned short>(0, k) = midval;
				}
				else if (temp_pixelk < (midval / 2))
				{
					aa.at<unsigned short>(0, k) = midval;
				}
				else
				{
					aa.at<unsigned short>(0, k) = temp_pixelk;
				}
			}
			// 高斯平滑
			Mat after_gauss;
			Mat before_gauss = src(Range(j, j + 1), Range(start[aL] - offset, end[aL] + offset + 1));
			after_gauss = smoothdata(before_gauss, 10);
			after_gauss.copyTo(src(Range(j, j + 1), Range(start[aL] - offset, end[aL] + offset + 1)));
		}
		finish2 = clock();
		double duration2 = (double)(finish2 - start2) / CLOCKS_PER_SEC;
		cout << "平滑时间： " << duration2 << endl;
		imwrite("gaosi.png", src);
		namedWindow("高斯平滑后的结果", CV_WINDOW_NORMAL);
		imshow("高斯平滑后的结果", src);
		waitKey(0);


		// 灰度直方图均衡化
		/*for (int m = 0; m < rows; m++)
		{
			for (int n = 0; n < cols; n++)
			{
				int pxv = src1.at<unsigned short>(m, n);
				pxv = pxv / 256;
				src1.at<unsigned short>(m, n) = int(pxv);
			}
		}*/
		/*src1 = bit16Tobit8(src1);
		src1.convertTo(src1, CV_8U);
		equalizeHist(src1, src1);
		namedWindow("直方图均衡化后的结果", CV_WINDOW_NORMAL);
		imshow("直方图均衡化后的结果", src1);
		waitKey(0);
		destroyAllWindows;*/
	}
	//return src;
}

int main()
{
	Mat src = imread("./images/png/d2.png", CV_LOAD_IMAGE_UNCHANGED);
	Mat result;
	int start[1] = { 257 };
	int end[1] = { 288 };
	int offset = 30;
	int line_nums = 1;
	renovation(src, start, end, line_nums, offset);
	/*namedWindow("修复结果", CV_WINDOW_NORMAL);
	imshow("修复结果", result);
	waitKey(0);
	destroyAllWindows;*/
	return 0;
}