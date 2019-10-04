#pragma once
#include<iostream>
#include"..\eigen3\Eigen\Dense"
#include <string>
#include <vector>  
#include <fstream>  
#include <sstream> 
#include <istream> 
using namespace std;
using namespace Eigen;

class DataGenerator
{
public:
	int dim, XY_train, XY_test;
	MatrixXd X_train;
	MatrixXd Y_train;
	MatrixXd X_test;
	MatrixXd Y_test;

private:
	int count = 0;

public:
	DataGenerator(int dim, int XY_train, int XY_test);
	bool load_housing_data(string path, int dim);
	bool load_wine_data(string path, int dim);
};



