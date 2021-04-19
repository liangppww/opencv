#include "MyOpen.h"
#include <QtWidgets/QApplication>

//#include<qlabel.h>
#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	

	MyOpen w;
	w.show();
	return a.exec();
}
