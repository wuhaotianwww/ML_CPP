#pragma once
#include<iostream>
#include<string>
#include"..\Utils\utils.h"
#include "..\eigen3\Eigen\Dense"

using namespace std;
using namespace Eigen;


class LinearRE
{
public:
	MatrixXd X_train;
	MatrixXd Y_train;
	MatrixXd X_test;
	MatrixXd Y_test;
	VectorXd theta;

private:
	int count = 0;

public:
	LinearRE(MatrixXd X_train, MatrixXd Y_train, MatrixXd X_test, MatrixXd Y_test);
	bool  fit();
	double loss_lss();
	VectorXd predect_lss();

	bool fit_gd(int loop, double lr);
	double loss_gd();
	VectorXd predect_gd();
};