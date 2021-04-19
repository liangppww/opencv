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
        Mat frame;//����һ����������ƵԴһ֡һ֡��ʾ
        cap >> frame;
		if (!cap.isOpened())
			return;
		vector<Rect> faces;//vector�������⵽��faces
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//ת�ҶȻ�����������
		cascada.detectMultiScale(frame_gray, faces, 1.1, 4,0, Size(70, 70), Size(1000, 1000));
	
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
		}

		
		
		if (faces.size() == 1)
		{
			Mat faceROI = frame_gray(faces[0]);//�ڻҶ�ͼ�н�Ȧ��������������ü���				 
			
			cv::resize(faceROI, myFace, Size(92, 112), 0, 0, INTER_LINEAR);
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 225), 2, 0);//�� faces[0].tl()�����Ͻ�����д���
			string filename = format("%d.jpg", pic_num); //����ڵ�ǰ��Ŀ�ļ�����1-10.jpg ������format����תΪ�ַ���
			imshow(filename, myFace);//��ʾ��size�����
			char key=waitKey(500);//�ȴ�500us
			switch (key)
			{
			case'p':
				pic_num++;//��ż�1
				imwrite(filename, myFace);
				destroyWindow(filename);//:����ָ���Ĵ���
				break;
			default:
				break;
			}
			
		}

		ui.stackedWidget->setCurrentIndex(0);
        QImage qimg = Mat2QImage(frame);
        ui.imagelabel1->setPixmap(QPixmap::fromImage(qimg));
        ui.imagelabel1->resize(ui.imagelabel1->pixmap()->size());
      
		//imshow("frame", frame);//��ʾ��Ƶ��
		if (waitKey(300) == 27) //Esc���˳���ESC��ASCLL��Ϊ27 
			break;
    }
	
	cap.release();
	destroyAllWindows();//�ر����д���

	return;

}

Mat MyOpen::norm_0_255(InputArray _src)
{
	Mat src = _src.getMat();
	// �����ͷ���һ����һ�����ͼ�����:  
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

	// 2�����������ͼ�����ݺͶ�Ӧ�ı�ǩ  
	vector<Mat> images;
	vector<int> labels;
	// ��ȡ����. ����ļ����Ϸ��ͻ����  
	// ������ļ����Ѿ�����.  
	try
	{
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// �ļ������⣬����ɶҲ�������ˣ��˳���  
		exit(1);
	}
	// ���û�ж�ȡ���㹻ͼƬ��Ҳ�˳�.  
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		//CV_Error(CV_StsError, error_message);
	}

	

	// ����ļ��д�������Ǵ�������ݼ����Ƴ����һ��ͼƬ  
	//[gm:��Ȼ������Ҫ�����Լ�����Ҫ�޸ģ���������˺ܶ�����]  
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
	// ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���  
	int predictedLabel = model->predict(testSample);
	//int predictedLabel1 = model1->predict(testSample);
	//int predictedLabel2 = model2->predict(testSample);

	// ����һ�ֵ��÷�ʽ�����Ի�ȡ���ͬʱ�õ���ֵ:  
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
	VideoCapture cap(0);    //��Ĭ������ͷ  
	if (!cap.isOpened())
	{
		return ;
	}
	Mat frame;
	Mat edges;
	Mat gray;

	CascadeClassifier cascade;
	bool stop = false;
	//ѵ���õ��ļ����ƣ������ڿ�ִ���ļ�ͬĿ¼��  
	cascade.load("haarcascade_frontalface_alt2.xml");

	Ptr<BasicFaceRecognizer> modelEigen = EigenFaceRecognizer::create();
	modelEigen->read("MyFacePCAModel.xml");
	while (1)
	{
		cap >> frame;
		//�������ڴ����������������  
		vector<Rect> faces(0);
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//�ı�ͼ���С��ʹ��˫���Բ�ֵ  
		//resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);  
		//�任���ͼ�����ֱ��ͼ��ֵ������  
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
			//����ͼ��Ӧ���ǻҶ�ͼ  
			predictPCA = modelEigen->predict(face_test);
		}

		cout << predictPCA << endl;
		string str;
		switch (predictPCA) //��ÿ������ʶ��
		{
		case 41:str = "handsom"; break;
		case 42:str = "beautifu"; break;
		
		default: str = "You look so ugly!"; break;
		}

		putText(frame, str, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));

		imshow("face", frame);
		if (waitKey(300) == 27) //Esc���˳���ESC��ASCLL��Ϊ27 
			break;
	}

	
	cap.release();
	destroyAllWindows();//�ر����д���
	return ;


}
