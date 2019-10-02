#include<iostream>
#include<algorithm>
#include<string>
#include"Utils\utils.h"
#include"model\LinearRE.h"
#include "eigen3\Eigen\Dense"

using namespace std;
using namespace Eigen;

int main()
{
	cout << "Hello World!" << endl;
	
	//׼������
	DataGenerator housingdata(13, 506, 506);
	housingdata.load_data_from_data("./data/housing.data", 13);
	//׼��ģ��
	LinearRE lr(housingdata.X_train, housingdata.Y_train, housingdata.X_test, housingdata.Y_test);
	//ʹ����С���˷�ѵ��
	lr.fit();
	cout << lr.theta << endl;
	lr.predect_lss();
	cout<< lr.loss_lss() << endl;
	//ʹ���ݶ��½���ѵ��
	lr.fit_gd(100000, 0.000000001);
	cout << lr.theta << endl;
	lr.predect_gd();
	cout << lr.loss_gd() << endl;


	system("pause");
	return 0;
}


/*
VectorXd:������
RowVectorXd��������
.transpose()ת��
.inverse() ����
.adjoint()�������
.determinant()����ʽ
.eigenvalues()����ֵ
.eigenvectors()��������
MatrixXd m = MatrixXd::Random(13, 1);              //�������3*3��double�;���
MatrixXd mm(14, 1);
mm << m, MatrixXd::Constant(1, 1, 1.0);      //MatrixXd::Constant(3,3,1.2)��ʾ����3*3��double�;��󣬸þ�������Ԫ�ؾ�Ϊ1.2
VectorXd v(14);    
v << -0.0877,  0.0496, - 0.0648,  0.0164, - 0.0025,  0.0755,  0.0374, - 0.1608,  0.1921, - 0.0138, - 0.2892,  0.0054, - 0.6843, 36.45948839;         // ������ֵ
*/