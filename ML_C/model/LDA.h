#pragma once
#include<iostream>
#include<string>
#include"..\Utils\utils.h"
#include "..\eigen3\Eigen\Dense"
using namespace std;
using namespace Eigen;

class LDA
{
public:
	MatrixXd X1_train;
	MatrixXd Y1_train;
	MatrixXd X2_train;
	MatrixXd Y2_train;
	MatrixXd X3_train;
	MatrixXd Y3_train;
	VectorXd w12;
	VectorXd w13;
	VectorXd w23;
	double w12_l1 = 0;
	double w12_l2 = 0;
	double w13_l1 = 0;
	double w13_l3 = 0;
	double w23_l2 = 0;
	double w23_l3 = 0;

private:
	int count = 0;

public:
	LDA(MatrixXd X_train, MatrixXd Y_train, int len);
	bool fit();
	double loss(MatrixXd X_train, MatrixXd Y_train, int len);
	VectorXd predect(MatrixXd X_train, int len);
};