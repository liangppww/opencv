#include "MyOpen.h"
#include <QDebug>

#include <iostream>  
#include <fstream>  
#include <sstream> 


//#include<opencv2\face\facerec.hpp>
//using namespace cv::face;




MyOpen::MyOpen(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);






}

QImage MyOpen::Mat2QImage(const cv::Mat &mat)
{
    switch (mat.type())
    {
        // 8-bit, 4 channel
    case CV_8UC4:
    {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32);
        return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }

    // 8-bit, 1 channel
    case CV_8UC1:
    {
        static QVector<QRgb>  sColorTable;
        // only create our color table once
        if (sColorTable.isEmpty())
        {
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        }
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }

    default:
        qDebug("Image format is not supported: depth=%d and %d channels\n", mat.depth(), mat.channels());
        break;
    }
    return QImage();
}

void MyOpen::on_getimage_clicked()
{
	/*VideoCapture cap0("test.mp4");
	if (!cap0.isOpened())
		return;*/
	ui.stackedWidget->setCurrentIndex(0);
	
	

	Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2(100, 25, 0);
	mog->setVarThreshold(20);
	mog->setNMixtures(2);

	Mat foreGround;
	Mat backGround;
	int trainCounter = 0;
	bool dynamicDetect = true;

	namedWindow("src");
	namedWindow("foreground");
	Mat src;
	bool stop = false;
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return;
	


    while (1)
    {
      /*  Mat frame;//定义一个变量把视频源一帧一帧显示
        cap >> frame;
		Mat frame_gray, foregroundFrame;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//转灰度化，减少运算
		imshow("frame_gray", frame_gray);
		
		mog->apply(frame_gray, foregroundFrame, -1);
		imshow("srcframe", frame);
		//cv::namedWindow("qianjing", WINDOW_NORMAL);
		imshow("desforegroundFrame", foregroundFrame); //

		//ui.stackedWidget->setCurrentIndex(0);
        QImage qimg = Mat2QImage(frame);
        ui.imagelabel1->setPixmap(QPixmap::fromImage(qimg));
        ui.imagelabel1->resize(ui.imagelabel1->pixmap()->size());*/

		cap >> src;
		if (dynamicDetect)
		{
			mog->apply(src, foreGround, 0.005);
			//图像处理过程
			medianBlur(foreGround, foreGround, 3);
			dilate(foreGround, foreGround, Mat(), Point(-1, -1), 3);
			erode(foreGround, foreGround, Mat(), Point(-1, -1), 6);
			dilate(foreGround, foreGround, Mat(), Point(-1, -1), 3);
			imshow("foreground", foreGround);
			if (trainCounter <50)//训练期间所得结果为不准确结果，不应作为后续
			{
				Mat findc;
				foreGround.copyTo(findc);
				vector<vector<Point>> contours;
				cv::findContours(findc, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

				//targets.clear();
				const int maxArea = 800;
				size_t s = contours.size();
				for (size_t i = 0; i < s; i++)
				{
					double area = abs(contourArea(contours[i]));
					if (area > maxArea)
					{
						Rect mr = boundingRect(Mat(contours[i]));
						rectangle(src, mr, Scalar(0, 0, 255), 2, 8, 0);
						//targets.push_back(mr);
					}
				}
				//string text;					
				char text[50];
				sprintf_s(text, "background training -%d- ...", trainCounter);
				putText(src, text, Point(50, 50), 3, 1, Scalar(0, 255, 255), 2, 8, false);
				//delete[] text;

			}
			else
			{
				//detects.clear();
				Mat findc;
				foreGround.copyTo(findc);
				vector<vector<Point>> contours;
				cv::findContours(findc, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
				const int maxArea = 500;
				size_t s = contours.size();
				RNG rng;
					for (size_t i = 0; i < s; i++)
					{
						double area = abs(contourArea(contours[i]));
						if (area > maxArea)
						{
							Scalar sca_color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
							Rect mr = boundingRect(Mat(contours[i]));
							rectangle(src, mr, sca_color, 2, 8, 0);

							//可以对动态目标进行相应操作

						}
					}

			}
			trainCounter++;
		}

		imshow("src", src);
      
		//imshow("frame", frame);//显示视频流
		if (waitKey(30) == 27) //Esc键退出，ESC的ASCLL码为27 
			break;
    }
	
	cap.release();
	destroyAllWindows();//关闭所有窗口

	return;

}

Mat MyOpen::norm_0_255(InputArray _src)
{
	Mat src = _src.getMat();
	// 创建和返回一个归一化后的图像矩阵:  
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

