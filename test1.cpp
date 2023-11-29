#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<iomanip>
#include<cmath>

using namespace cv;
using namespace std;

void Normalize(Mat m)
{
	//% I1 = (K1 - min(min(K1))). / (max(max(K1)) - min(min(K1)));
	//% I2 = (K21 - min(min(K21))). / (max(max(K21)) - min(min(K21)));
	double mx, mn;
	Point pmx, pmn;
	minMaxLoc(m, &mn, &mx, &pmn, &pmx);
	m = (m - mn) / (mx - mn);
}
// 图像取反操作
Mat invert_pixel(Mat src)
{
	Mat dst = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int pixval = src.at<unsigned char>(i, j);
			dst.at<unsigned char>(i, j) = 255 - pixval;
		}
	}
	return dst;
}

Mat conv2D(Mat src)
{	
	int borderType = BORDER_DEFAULT;
	Point anchor = Point(-1, -1);
	Mat kernel = (Mat_<float>(3, 3) << 0.4444, 0.1111, 0.4444, 0.1111, -2.2222, 0.1111, 0.4444, 0.1111, 0.4444);
	Mat kernelFilp;
	Mat dst;
	Mat res;
	flip(kernel, kernelFilp, -1);
	filter2D(src, dst, -1, kernel, anchor, 0.0, borderType);
	//res = dst + 100;
	return dst;
}

// 寻找特定面积的轮廓
Mat bwareaopen(Mat src)
{
	double area;
	Mat temp = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	Mat dst = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	/*Mat res = src;*/
	vector<vector<Point>>contours;
	vector<vector<Point>>::iterator itr; //轮廓迭代器
	findContours(src,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	itr = contours.begin();
	while (itr != contours.end())
	{
		area = contourArea(*itr);
		//cout << area << endl;
		if (area < 28000)
		{
			itr = contours.erase(itr);
		}
		else
		{
			++itr;
		}
	}
	drawContours(temp, contours, -1, Scalar(255), -1);// CV_FILLED
	bitwise_and(src, temp, dst);
	return dst;
}

Mat bwareaopen1(Mat src)
{
	double area;
	Mat temp = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	Mat dst = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	/*Mat res = src;*/
	vector<vector<Point>>contours;
	vector<vector<Point>>::iterator itr; //轮廓迭代器
	findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	itr = contours.begin();
	while (itr != contours.end())
	{
		area = contourArea(*itr);
		//cout << area << endl;
		if (area < 10000)
		{
			itr = contours.erase(itr);
		}
		else
		{
			++itr;
		}
	}
	drawContours(temp, contours, -1, Scalar(255), -1);// CV_FILLED
	bitwise_and(src, temp, dst);
	return dst;
}


Mat conv2D1(Mat src)
{
	int borderType = BORDER_DEFAULT;
	Point anchor = Point(-1, -1);
	Mat kernel = (Mat_<int>(5, 5) << 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, 16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0);
	Mat kernelFilp;
	Mat dst;
	flip(kernel, kernelFilp, -1);
	filter2D(src, dst, -1, kernelFilp, anchor, 0.0, borderType);
	return dst;
}

//寻找阈值
void retchlim_custom(Mat src,int in_low_high[])
{
	int rows = src.rows;
	int cols = src.cols;
	double numpixels = src.total();
	int hist[256] = { 0 };  // 灰度直方图
	int pixval;
	for (int i = 0; i <	rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			pixval = src.at<unsigned char>(i,j);
			hist[pixval] += 1;
		}
	}
	// 找到底部和顶部1%的像素值
	double summary[256] = { 0 }; // 灰度直方图累加
	for (int k = 0; k < 256; k++)
	{
		int temp = k;
		double s = 0;
		while (temp >= 0)
		{
			s = s + hist[temp];
			temp--;
		}
		summary[k] = s;
	}
	float cdf[256] = { 0 }; // 像素值所占比例
	for (int m = 0; m < 256; m++)
	{	
		cdf[m] = summary[m] / numpixels;
	}
	// 大阈值和小阈值
	int MaxVal = 0;
	int MinVal = 0;
	/*int in_low_high[2];*/
	for (int n = 0; n < 256; n++)
	{	
		int temp1 = n;
		if ((cdf[n] <= 0.01) && (cdf[n + 1] > 0.01))
		{
			MinVal = temp1;
		}
		else if ((cdf[n] > 0.99) && (cdf[n - 1] <= 0.99))
		{
			MaxVal = temp1;
		}		
	}
	in_low_high[0] = MinVal;
	in_low_high[1] = MaxVal;
}

// 灰度值自适应函数
void adjust_custom(Mat src, Mat dst, int minval, int maxval)
{
	int out_low = 0;
	int out_high = 255;
	for (int i = 0; i < src.rows; i++) 
	{
		for (int j = 0; j < src.cols; j++)
		{
			int pixelval = src.at<unsigned char>(i, j);
			//cout << "原图的像素值： " << pixelval << endl;
			if (pixelval <= minval)
			{
				dst.at<unsigned char>(i, j) = out_low;
				//cout << "目标像素值： " << dst.at<unsigned char>(i, j) << endl;
			}
			else if (pixelval >= maxval)
			{
				dst.at<unsigned char>(i, j) = out_high;
				//cout << "目标像素值： " << (int)dst.at<unsigned char>(i, j) << endl;
			}
			else
			{	
				int pix = src.at<unsigned char>(i, j);
				dst.at<unsigned char>(i, j) = 0.95*pix;
				// cout << "目标像素值： " << (int)dst.at<unsigned char>(i, j) << endl;
			}	
		}
	}
}

Mat imclearBorder(Mat src,int radius=8)
{
	Mat dst1 = src;
	Mat dst2 = Mat::zeros(Size(src.cols, src.rows), CV_8U);
	//找到图像轮廓
	vector<vector<Point>>contours;
	findContours(src, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	// 获取图像尺寸
	int rows = src.rows;
	int cols = src.cols;
	// 对每个轮廓进行迭代，
	// 查看轮廓中的每一个点
	vector<int>contouridx;   // 定义要删除的轮廓的索引向量
	for (int i = 0; i < contours.size(); i++)
	{
		// 得到轮廓
		vector<Point>cnt;
		cnt = contours[i];
		// 查看轮廓中每一个点
		for (int j = 0; j < cnt.size(); j++)
		{
			int x = cnt[j].x;
			int y = cnt[j].y;
			// 如果在边界的指定半径内部，则删除轮廓
			bool check1 = (x >= 0 && x < radius) || (x >= rows - 1 - radius && x < rows);
			bool check2 = (y >= 0 && y < radius) || (y >= cols - 1 - radius && y < cols);
			if (check1 || check2)
			{
				contouridx.push_back(j);
			}
		}
	}
	if (contouridx.size() <= 0)
	{
		cout << "没有要去除的边" << endl;
		return dst1;
	}	
	else
	{	
		cout << contouridx.size() << endl;
		for (int k = 0; k < contouridx.size(); k++)
		{
			drawContours(dst2, contours, contouridx[k], Scalar(255), 5);
		}
		double area;
		Mat temp = Mat::zeros(Size(dst2.cols, dst2.rows), CV_8U);
		Mat dst = Mat::zeros(Size(dst2.cols, dst2.rows), CV_8U);
		vector<vector<Point>>contours1;
		vector<vector<Point>>::iterator itr; //轮廓迭代器
		findContours(dst2, contours1, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		itr = contours1.begin();
		while (itr != contours1.end())
		{
			area = contourArea(*itr);
			//cout << area << endl;
			if (area > 1000)
			{
				itr = contours1.erase(itr);
			}
			else
			{
				++itr;
			}
		}
		drawContours(temp, contours1, -1, Scalar(255), -1);
		return temp;
	}
}

// 轮廓重心函数
vector<Point> zhongxin(Mat src,Mat dst,vector<Point>coordinate)
{
	//寻找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy1;
	findContours(src, contours, hierarchy1, RETR_LIST, CHAIN_APPROX_NONE, Point());
	//获取某一轮廓重心点，并在图像中绘制
	cvtColor(src, src, COLOR_GRAY2BGR);
	for (int i = 0; i < contours.size(); i++)
	{
		Moments M;
		M = moments(contours[i]); // 计算图像轮廓的矩
		float cX = double(M.m10 / M.m00);  // 求取轮廓重心的X坐标
		float cY = double(M.m01 / M.m00);
		Point point; // 重心坐标点
		point.x = cX;
		point.y = cY;
		coordinate.push_back(point);
		//显示轮廓重心并提取坐标点
		circle(src, Point2d(cX, cY), 1, Scalar(0, 0, 255), 5, 8);
		//putText(dst, "center", Point2d(cX - 20, cY - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, 8);
		cout << "重心坐标：" << cX << ' ' << cY << endl << endl;
	}
	/*dst = src;
	namedWindow("重心", CV_NORMAL);
	imshow("重心", dst);
	waitKey(0);*/
	cout << coordinate << endl;
	return coordinate;
}

// 拟合直线
Vec4f jiaodu(vector<Point>coordinate)
{
	Vec4f lines;
	double param = 0.0;
	double reps = 0.01;
	double aeps = 0.01;
	fitLine(coordinate, lines, DIST_L1, param, reps, aeps);
	return lines;
}

// 寻找绘图点
vector<Point> polyfit(Mat a,vector<Point>coordinate)
{
	Vec4f lines = jiaodu(coordinate);
	float k = lines[1] / lines[0]; // 直线斜率
	// 绘制拟合直线图，直线点斜式方程:y-line[3]=(line[1]/line[0]) * (x - line[2])
	Point point1;
	Point point2;
	cvtColor(a, a, COLOR_GRAY2BGR);
	double coorY = k * (1600 - lines[2]) + lines[3]; 
	double coorX = (1200 - lines[3]) / k + lines[2];
	point1.x = 1600;
	point1.y = coorY;
	point2.x = coorX;
	point2.y = 1200;

	vector<Point>coord;
	coord.push_back(point1);  // 绘制直线的坐标
	coord.push_back(point2);
	line(a, point1, point2, Scalar(0, 0, 255), 5, 8);
	namedWindow("拟合直线", CV_NORMAL);
	imshow("拟合直线", a);
	waitKey(0);

	return coord;
}

// 角度差异
void angle(vector<Point>coordinate1, vector<Point>coordinate2)
{
	Vec4f lines1;
	Vec4f lines2;
	lines1 = jiaodu(coordinate1);
	lines2 = jiaodu(coordinate2);
	float k1 = lines1[1] / lines1[0]; // 直线斜率
	float angle1 = atan(k1); // 弧度
	double jiaodu1 = angle1 * 180 / 3.14;

	float k2 = lines2[1] / lines2[0]; // 直线斜率
	float angle2 = atan(k2); // 弧度
	double jiaodu2 = angle2 * 180 / 3.14;

	double registration = jiaodu1 - jiaodu2;
	cout << "最终结果： " << registration << endl;
}

void plot(vector<Point>coordinate1, vector<Point>coordinate2)
{
	Mat result = Mat::zeros(Size(1600, 1200), CV_8U(3));
	for (int i = 0; i < 1200; i++)
	{
		for (int j = 0; j < 1600; j++)
		{
			result.at<unsigned char>(i, j) = 255;
		}
	}
	cvtColor(result, result, COLOR_GRAY2BGR);
	line(result, coordinate1[0], coordinate1[1], Scalar(0, 0, 255), 5, 8);
	line(result, coordinate2[0], coordinate2[1], Scalar(0, 255, 0), 5, 8);
	//imwrite("fitlines.png", result);
	namedWindow("结果", CV_WINDOW_NORMAL);
	imshow("结果", result);
	waitKey(0);
}

//伽马变换
void gamma_translation(Mat image,int gamma)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float pixval = image.at<unsigned char>(i, j);
			pixval = pixval / 255;
			image.at<unsigned char>(i, j) = int(pow(pixval, gamma)*255.0);
			cout << pow(pixval, gamma) << endl;
		}
	}
}

// 二值化函数
void mianzaoerzhihua(Mat A,Mat B,Mat dst[]) {
	// 拉普拉斯卷积
	Mat dst1;
	Mat dst2;
	Mat K1;
	Mat K2;
	/*Laplacian(A, dst1, CV_8U, 3);
	Laplacian(B, dst2, CV_8U, 3);*/
	dst1 = conv2D(A);
	dst2 = conv2D(B);
	//namedWindow("拉普拉斯1", CV_WINDOW_NORMAL);
	//imshow("拉普拉斯1", dst1);
	//namedWindow("拉普拉斯2", CV_WINDOW_NORMAL);
	//imshow("拉普拉斯2", dst2);
	//waitKey(0);
	//差分
	K1 = A - dst1;  // [1600 X 1200]
	K2 = B - dst2; 
	/*namedWindow("差分1", CV_WINDOW_NORMAL);
	imshow("差分1", K1);
	namedWindow("差分2", CV_WINDOW_NORMAL);
	imshow("差分2", K2);
	waitKey(0);*/	
	Mat pingyu1 = Mat::ones(1200, 1600, CV_8U);
	Mat pingyu2 = Mat::ones(1200, 1600, CV_8U);
	int in_low_high1[2];
	int in_low_high2[2];
	retchlim_custom(K1, in_low_high1);
	adjust_custom(K1, pingyu1,in_low_high1[0], in_low_high1[1]);
	retchlim_custom(K2, in_low_high2);
	adjust_custom(K2, pingyu2, in_low_high2[0], in_low_high2[1]);
	//namedWindow("res1", CV_WINDOW_NORMAL);
	//imshow("res1", pingyu1);
	//namedWindow("res2", CV_WINDOW_NORMAL);
	//imshow("res2", pingyu2);
	//waitKey(0);
	Normalize(pingyu1);
	Normalize(pingyu2);
	Mat J1 = conv2D1(pingyu1);
	Mat J2 = conv2D1(pingyu2);
	/*for (int i = 0; i < J1.rows; i++)
	{
		for (int j = 0; j < J1.cols; j++)
		{
			cout << (int)J1.at<unsigned char>(i, j) << endl;
		}
	}*/
	/*namedWindow("频域滤波1", CV_WINDOW_NORMAL);
	imshow("频域滤波1", J1);
	namedWindow("频域滤波2", CV_WINDOW_NORMAL);
	imshow("频域滤波2", J2);
	waitKey(0);*/

	// 自动阈值分割
	Mat res1;
	Mat res2;
	threshold(J1, res1, 0, 255, THRESH_OTSU + THRESH_BINARY);  
	threshold(J2, res2, 0, 255, THRESH_OTSU + THRESH_BINARY);
	/*namedWindow("二值1", CV_WINDOW_NORMAL);
	imshow("二值1", res1);
	namedWindow("二值2", CV_WINDOW_NORMAL);
	imshow("二值2", res2);
	waitKey(0);*/
	//// 去除二值图像中连通面积小于28000的部分
	Mat ka1;
	Mat ka2;
	ka1 = bwareaopen(res1);
	ka2 = bwareaopen(res2);
	/*namedWindow("去除小面积1", CV_WINDOW_NORMAL);
	imshow("去除小面积1", ka1);
	namedWindow("去除小面积2", CV_WINDOW_NORMAL);
	imshow("去除小面积2", ka2);
	waitKey(0);*/
	/*K1.convertTo(K1, CV_8U);
	K2.convertTo(K2, CV_8U);*/
	// 对图像取反操作
	Mat inv1;
	Mat inv2;
	inv1 = invert_pixel(ka1);
	inv2 = invert_pixel(ka2);
	/*namedWindow("取反1", CV_WINDOW_NORMAL);
	imshow("取反1", ka1);
	namedWindow("取反2", CV_WINDOW_NORMAL);
	imshow("取反2", ka2);
	waitKey(0);*/
	dst[0] = inv1;
	dst[1] = inv2;
}

int main(int arg, char** argv) {
	Mat A = imread("./images/test1.bmp", -1); 
	if (A.empty())
		return -1;
	Mat B = imread("./images/test2.bmp", -1);
	if (B.empty())
		return -1;
	namedWindow("base", CV_WINDOW_NORMAL);
	imshow("base", A);
	namedWindow("rotation", CV_WINDOW_NORMAL);
	imshow("rotation", B);
	waitKey(0);
	// 对比度调整
	/*gamma_translation(A, 0.5);
	gamma_translation(B, 0.5);*/
	namedWindow("gamma1", CV_WINDOW_NORMAL);
	imshow("gamma1", A);
	namedWindow("gamma2", CV_WINDOW_NORMAL);
	imshow("gamma2", B);
	waitKey(0);

	// 二值化图像
	Mat erzhihua[2];
	mianzaoerzhihua(A, B, erzhihua);
	//创建结构元素并进行形态学处理
	Mat element1 = getStructuringElement(MORPH_RECT, Size(15, 15)); // 15
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));  // 5
	Mat pengzhang1;
	Mat pengzhang2;
	
	dilate(erzhihua[0], pengzhang1, element1); // 结构元素对图像进行膨胀操作
	dilate(erzhihua[1], pengzhang2, element1);
	namedWindow("膨胀1", CV_WINDOW_NORMAL);
	imshow("膨胀1", pengzhang1);
	namedWindow("膨胀2", CV_WINDOW_NORMAL);
	imshow("膨胀2", pengzhang1);
	waitKey(0);
	Mat close1;
	Mat close2;
	morphologyEx(pengzhang1, close1, MORPH_CLOSE, element2); // 进行闭运算
	morphologyEx(pengzhang2, close2, MORPH_CLOSE, element2);

	// 去除左上角的轮廓,保留最大的轮廓
	/*close1 = bwareaopen1(close1);
	close2 = bwareaopen1(close2);
*/

	namedWindow("闭运算1", CV_WINDOW_NORMAL);
	imshow("闭运算1", close1);
	namedWindow("闭运算2", CV_WINDOW_NORMAL);
	imshow("闭运算2", close2);
	waitKey(0);

	// 图像取反操作
	Mat inv1;
	Mat inv2;
	inv1 = invert_pixel(close1);
	inv2 = invert_pixel(close2);
	namedWindow("反色1", CV_WINDOW_NORMAL);
	imshow("反色1", inv1);
	namedWindow("反色2", CV_WINDOW_NORMAL);
	imshow("反色2", inv2);
	waitKey(0);
	// 边界对象抑制
	Mat bianjie1;
	Mat bianjie2;
	bianjie1 = imclearBorder(inv1);
	bianjie2 = imclearBorder(inv2);
	namedWindow("第一次边界抑制1", CV_WINDOW_NORMAL);
	imshow("第一次边界抑制1", bianjie1);
	namedWindow("第一次边界抑制2", CV_WINDOW_NORMAL);
	imshow("第一次边界抑制2", bianjie2);
	waitKey(0);
	// 再反色再边界抑制
	Mat inv12;
	Mat inv22;
	inv12 = invert_pixel(bianjie1);
	inv22 = invert_pixel(bianjie2);
	namedWindow("反色12", CV_WINDOW_NORMAL);
	imshow("反色12", inv12);
	namedWindow("反色22", CV_WINDOW_NORMAL);
	imshow("反色22", inv22);
	waitKey(0);
	Mat bianjie12;
	Mat bianjie22;
	bianjie12 = imclearBorder(inv12);
	bianjie22 = imclearBorder(inv22); 
	namedWindow("第二次边界抑制1", CV_WINDOW_NORMAL);
	imshow("第二次边界抑制1", bianjie12);
	namedWindow("第二次边界抑制2", CV_WINDOW_NORMAL);
	imshow("第二次边界抑制2", bianjie22);
	waitKey(0);

	// 找到质心并拟合直线
	Mat zhixin1;
	Mat zhixin2;
	vector<Point>coordinate1;
	vector<Point>coordinate2;
	coordinate1 = zhongxin(bianjie12,zhixin1, coordinate1);
	coordinate2 = zhongxin(bianjie22,zhixin2, coordinate2);
	// 拟合直线
	/*double jiaodu1;
	double jiaodu2;*/
	double registation;
	vector<Point>coord1;
	vector<Point>coord2;
	coord1 = polyfit(A,coordinate1);
	coord2 = polyfit(B,coordinate2);
	//plotline(A, coordinate1);
	//plotline(B, coordinate2);
	plot(coord1, coord2);
	angle(coordinate1, coordinate2);
	
}
