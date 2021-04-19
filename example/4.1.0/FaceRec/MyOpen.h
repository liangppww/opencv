#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MyOpen.h"

/*
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
*/
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

//using namespace cv::face;



class MyOpen : public QMainWindow
{
	Q_OBJECT

public:
	MyOpen(QWidget *parent = Q_NULLPTR);

    QImage Mat2QImage(const cv::Mat &mat);
	Mat norm_0_255(InputArray _src);
	void read_csv(string filename, vector<Mat>& images, vector<int>& labels, char separator );

	private slots:
    void on_getimage_clicked(); 
	void on_train_clicked();
	void on_predictface_clicked();


private:
	Ui::MyOpenClass ui;


};
