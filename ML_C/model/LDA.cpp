#include"LDA.h"

/********模型使用方法*********
//准备数据
DataGenerator winedata(13, 178, 178);
winedata.load_wine_data("./data/wine.data", 13);
//准备模型
LDA  model(winedata.X_train, winedata.Y_train, 178);
//拟合
model.fit();

double loss = model.loss(winedata.X_train, winedata.Y_train, 178);
cout << loss << endl;

******************************/

LDA::LDA(MatrixXd X_train, MatrixXd Y_train, int len)
{
	int x1=0, x2=0, x3 = 0;
	for (int i = 0; i < len; ++i)
	{
		if (Y_train(i, 0) < 1.5) x1++;
		else if (Y_train(i, 0) > 2.5) x3++;
		else x2++;
	}
	MatrixXd Mx1(x1, 13);
	MatrixXd Mx2(x2, 13);
	MatrixXd Mx3(x3, 13);
	MatrixXd My1(x1, 1);
	MatrixXd My2(x2, 1);
	MatrixXd My3(x3, 1);
	x1 = 0; x2 = 0; x3 = 0;
	for (int i = 0; i < len; ++i)
	{
		if (Y_train(i, 0) < 1.5)
		{
			Mx1.row(x1) =  X_train.row(i);
			My1.row(x1++) = Y_train.row(i);
		}
		else if (Y_train(i, 0) > 2.5)
		{
			Mx3.row(x3) = X_train.row(i);
			My3.row(x3++) = Y_train.row(i);
		}
		else
		{
			Mx2.row(x2) = X_train.row(i);
			My2.row(x2++) = Y_train.row(i);
		}
	}
	this->X1_train = Mx1.transpose();
	this->Y1_train = My1.transpose();
	this->X2_train = Mx2.transpose();
	this->Y2_train = My2.transpose();
	this->X3_train = Mx3.transpose();
	this->Y3_train = My3.transpose();
}
bool LDA::fit()
{
	int dim = this->X1_train.rows();
	VectorXd u1(dim);
	VectorXd u2(dim);
	VectorXd u3(dim);
	for (int i = 0; i < dim; ++i)
	{
		u1(i) = this->X1_train.row(i).sum() / this->X1_train.cols();
		u2(i) = this->X2_train.row(i).sum() / this->X2_train.cols();
		u3(i) = this->X3_train.row(i).sum() / this->X3_train.cols();
	}
	MatrixXd X1 = this->X1_train;
	MatrixXd X2 = this->X2_train;
	MatrixXd X3 = this->X3_train;
	for (int i = 0; i < X1.cols(); ++i)
		X1.col(i) = X1.col(i) - u1;
	for (int i = 0; i < X2.cols(); ++i)
		X2.col(i) = X2.col(i) - u2;
	for (int i = 0; i < X3.cols(); ++i)
		X3.col(i) = X3.col(i) - u3;
	MatrixXd sum1 = X1 * X1.transpose();
	MatrixXd sum2 = X2 * X2.transpose();
	MatrixXd sum3 = X3 * X3.transpose();
	this->w12 = (sum1 + sum2).inverse()*(u1 - u2);
	this->w13 = (sum1 + sum3).inverse()*(u1 - u3);
	this->w23 = (sum2 + sum3).inverse()*(u2 - u3);
	this->w12_l1 = this->w12.dot(u1);
	this->w12_l2 = this->w12.dot(u2);
	this->w13_l1 = this->w13.dot(u1);
	this->w13_l3 = this->w13.dot(u3);
	this->w23_l2 = this->w23.dot(u2);
	this->w23_l3 = this->w23.dot(u3);
	return true;
}
double LDA::loss(MatrixXd X_train, MatrixXd Y_train, int len)
{
	double ans = 0;
	VectorXd y = this->predect(X_train, len);
	VectorXd yy(len);
	yy << Y_train.col(0);
	ans = (yy - y).dot((yy - y));
	return ans;
}
VectorXd LDA::predect(MatrixXd X_train, int len)
{
	vector<double> ans;
	for (int i = 0; i < len; ++i)
	{
		double len1 = (X_train.row(i) * this->w12)(0, 0);
		if (abs(len1 - this->w12_l1) < abs(len1 - this->w12_l2))
		{
			double len3 = (X_train.row(i) * this->w13)(0, 0);
			if (abs(len3 - this->w13_l1) < abs(len3 - this->w13_l3))
				ans.push_back(1.0);
			else
				ans.push_back(3.0);
		}
		else 
		{
			double len2 = (X_train.row(i) * this->w23)(0, 0);
			if (abs(len2 - this->w23_l2) < abs(len2 - this->w23_l3))
				ans.push_back(2.0);
			else
				ans.push_back(3.0);
		}
	}
	return VectorXd::Map(&ans[0], len);
}