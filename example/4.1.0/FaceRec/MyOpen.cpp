#include "MyOpen.h"
#include <QDebug>

#include <iostream>  
#include <fstream>  
#include <sstream>  
//#include <math.h> 
#include<opencv2\face\facerec.hpp>
//#include<opencv2\face.hpp>
using namespace cv::face;




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
		return;
	ui.stackedWidget->setCurrentIndex(0);*/
	
	
	CascadeClassifier cascada;
	cascada.load("haarcascade_frontalface_alt2.xml");
	Mat frame, myFace;
	int pic_num = 1;
	
	
	VideoCapture cap;
    cap.open(0);
    while (1)
    {
        Mat frame;//定义一个变量把视频源一帧一帧显示
        cap >> frame;
		if (!cap.isOpened())
			return;
		vector<Rect> faces;//vector容器存检测到的faces
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//转灰度化，减少运算
		cascada.detectMultiScale(frame_gray, faces, 1.1, 4,0, Size(70, 70), Size(1000, 1000));
	
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
		}

		
		
		if (faces.size() == 1)
		{
			Mat faceROI = frame_gray(faces[0]);//在灰度图中将圈出的脸所在区域裁剪出				 
			
			cv::resize(faceROI, myFace, Size(92, 112), 0, 0, INTER_LINEAR);
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 225), 2, 0);//在 faces[0].tl()的左上角上面写序号
			string filename = format("%d.jpg", pic_num); //存放在当前项目文件夹以1-10.jpg 命名，format就是转为字符串
			imshow(filename, myFace);//显示下size后的脸
			char key=waitKey(500);//等待500us
			switch (key)
			{
			case'p':
				pic_num++;//序号加1
				imwrite(filename, myFace);
				destroyWindow(filename);//:销毁指定的窗口
				break;
			default:
				break;
			}
			
		}

		ui.stackedWidget->setCurrentIndex(0);
        QImage qimg = Mat2QImage(frame);
        ui.imagelabel1->setPixmap(QPixmap::fromImage(qimg));
        ui.imagelabel1->resize(ui.imagelabel1->pixmap()->size());
      
		//imshow("frame", frame);//显示视频流
		if (waitKey(300) == 27) //Esc键退出，ESC的ASCLL码为27 
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

void MyOpen::read_csv( string filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
		std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		//CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}

}





void MyOpen::on_train_clicked()
{
	ui.stackedWidget->setCurrentIndex(1);

	string fn_csv = "liang.txt";

	// 2个容器来存放图像数据和对应的标签  
	vector<Mat> images;
	vector<int> labels;
	// 读取数据. 如果文件不合法就会出错  
	// 输入的文件名已经有了.  
	try
	{
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// 文件有问题，我们啥也做不了了，退出了  
		exit(1);
	}
	// 如果没有读取到足够图片，也退出.  
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		//CV_Error(CV_StsError, error_message);
	}

	

	// 下面的几行代码仅仅是从你的数据集中移除最后一张图片  
	//[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	


	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	/*
	Ptr<BasicFaceRecognizer> model1 = FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");
	Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
	model2->train(images, labels);
	model2->save("MyFaceLBPHModel.xml");
	*/
	// 下面对测试图像进行预测，predictedLabel是预测标签结果  
	int predictedLabel = model->predict(testSample);
	//int predictedLabel1 = model1->predict(testSample);
	//int predictedLabel2 = model2->predict(testSample);

	// 还有一种调用方式，可以获取结果同时得到阈值:  
	//      int predictedLabel = -1;  
	//      double confidence = 0.0;  
	//      model->predict(testSample, predictedLabel, confidence);  

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	

	//getchar();
	//waitKey(0);
	//return ;

}
void MyOpen::on_predictface_clicked()
{
	ui.stackedWidget->setCurrentIndex(1);
	VideoCapture cap(0);    //打开默认摄像头  
	if (!cap.isOpened())
	{
		return ;
	}
	Mat frame;
	Mat edges;
	Mat gray;

	CascadeClassifier cascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下  
	cascade.load("haarcascade_frontalface_alt2.xml");

	Ptr<BasicFaceRecognizer> modelEigen = EigenFaceRecognizer::create();
	modelEigen->read("MyFacePCAModel.xml");
	while (1)
	{
		cap >> frame;
		//建立用于存放人脸的向量容器  
		vector<Rect> faces(0);
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//改变图像大小，使用双线性差值  
		//resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);  
		//变换后的图像进行直方图均值化处理  
		equalizeHist(gray, gray);

		cascade.detectMultiScale(gray, faces,1.1, 4,0,Size(30, 30), Size(500, 500));
		
		Mat face;
		Point text_lb;

		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				face = gray(faces[i]);
				text_lb = Point(faces[i].x, faces[i].y);
				rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);
			}
		}

		Mat face_test;

		int predictPCA = 0;
		if (face.rows >= 120)
		{
			cv::resize(face, face_test, Size(92, 112));

		}
		
		if (!face_test.empty())
		{
			//测试图像应该是灰度图  
			predictPCA = modelEigen->predict(face_test);
		}

		cout << predictPCA << endl;
		string str;
		switch (predictPCA) //对每张脸都识别
		{
		case 41:str = "handsom"; break;
		case 42:str = "beautifu"; break;
		
		default: str = "You look so ugly!"; break;
		}

		putText(frame, str, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));

		imshow("face", frame);
		if (waitKey(300) == 27) //Esc键退出，ESC的ASCLL码为27 
			break;
	}

	
	cap.release();
	destroyAllWindows();//关闭所有窗口
	return ;


}
