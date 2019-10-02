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
	
	//准备数据
	DataGenerator housingdata(13, 506, 506);
	housingdata.load_data_from_data("./data/housing.data", 13);
	//准备模型
	LinearRE lr(housingdata.X_train, housingdata.Y_train, housingdata.X_test, housingdata.Y_test);
	//使用最小二乘法训练
	lr.fit();
	cout << lr.theta << endl;
	lr.predect_lss();
	cout<< lr.loss_lss() << endl;
	//使用梯度下降法训练
	lr.fit_gd(100000, 0.000000001);
	cout << lr.theta << endl;
	lr.predect_gd();
	cout << lr.loss_gd() << endl;


	system("pause");
	return 0;
}


/*
VectorXd:列向量
RowVectorXd：行向量
.transpose()转置
.inverse() 求逆
.adjoint()伴随矩阵
.determinant()行列式
.eigenvalues()特征值
.eigenvectors()特征向量
MatrixXd m = MatrixXd::Random(13, 1);              //随机生成3*3的double型矩阵
MatrixXd mm(14, 1);
mm << m, MatrixXd::Constant(1, 1, 1.0);      //MatrixXd::Constant(3,3,1.2)表示生成3*3的double型矩阵，该矩阵所有元素均为1.2
VectorXd v(14);    
v << -0.0877,  0.0496, - 0.0648,  0.0164, - 0.0025,  0.0755,  0.0374, - 0.1608,  0.1921, - 0.0138, - 0.2892,  0.0054, - 0.6843, 36.45948839;         // 向量赋值
*/