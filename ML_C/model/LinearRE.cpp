#include"LinearRE.h"


/********模型使用方法*********

//准备数据
DataGenerator housingdata(13, 506, 506);
housingdata.load_housing_data("./data/housing.data", 13);
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

******************************/






LinearRE::LinearRE(MatrixXd X_train, MatrixXd Y_train, MatrixXd X_test, MatrixXd Y_test)
{
	this->X_train = X_train;
	this->Y_train = Y_train;
	this->X_test = X_test;
	this->Y_test = Y_test;
}

bool  LinearRE::fit()
{
	MatrixXd train_x(14, 506);
	train_x << this->X_train.transpose(), MatrixXd::Constant(1, 506, 1.0);
	this ->theta = (train_x * train_x.transpose()).inverse() * train_x * this->Y_train;
	return true;
}
double LinearRE::loss_lss()
{
	MatrixXd test_x(14, 506);
	test_x << this->X_test.transpose(), MatrixXd::Constant(1, 506, 1.0);
	VectorXd yy = test_x.transpose() * this->theta - this->Y_test;
	RowVectorXd out = (yy.transpose() * yy) / 1012.0;
	return out(0,0);
}
VectorXd LinearRE::predect_lss()
{
	MatrixXd train_x(14, 506);
	train_x << this->X_train.transpose(), MatrixXd::Constant(1, 506, 1.0);
	return train_x.transpose() * this->theta;
}

bool LinearRE::fit_gd(int loop, double lr) {
	this->theta = VectorXd::Zero(14);
	MatrixXd train_x(14, 506);
	train_x << this->X_train.transpose(), MatrixXd::Constant(1, 506, 1.0);
	for (int i = 0; i < loop; ++i)
	{
		VectorXd grad = ((train_x * (train_x.transpose()* this->theta - this->Y_train))) *lr;
		//cout << grad << endl;
		this->theta = this->theta - grad;
	}
	return true;
}
double LinearRE::loss_gd()
{
	MatrixXd test_x(14, 506);
	test_x << this->X_test.transpose(), MatrixXd::Constant(1, 506, 1.0);
	VectorXd yy = test_x.transpose() * this->theta - this->Y_test;
	RowVectorXd out = (yy.transpose() * yy) / 1012.0;
	return out(0, 0);
}
VectorXd LinearRE::predect_gd()
{
	MatrixXd train_x(14, 506);
	train_x << this->X_train.transpose(), MatrixXd::Constant(1, 506, 1.0);
	return train_x.transpose() * this->theta;
}
