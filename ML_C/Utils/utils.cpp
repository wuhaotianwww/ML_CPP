#include "utils.h"

DataGenerator::DataGenerator(int dim, int XY_train, int XY_test)
{
	this->dim = dim;
	this->XY_train = XY_train;
	this->XY_test = XY_test;
	this->X_train =  MatrixXd(XY_train, dim);
	this->Y_train = MatrixXd(XY_train, 1);
	this->X_test = MatrixXd(XY_test,dim);
	this->Y_test = MatrixXd(XY_test, 1);
}

bool DataGenerator::load_data_from_data(string path, int dim)
{
	ifstream inFile(path, ios::in);
	string lineStr;
	vector<vector<double>> strArray;
	while (getline(inFile, lineStr))
	{
		// 存成二维表结构  
		istringstream sin(lineStr);
		double str;
		vector<double> lineArray;
		// 按照逗号分隔  
		while (sin>>str)
			lineArray.push_back(str);
		strArray.push_back(lineArray);
	}
	for (int i = 0; i < strArray.size(); i++) {
		this->X_train.row(i) = VectorXd::Map(&strArray[i][0], dim);
		this->X_test.row(i) = VectorXd::Map(&strArray[i][0], dim);
		this->Y_train.row(i) = VectorXd::Map(&strArray[i][strArray[i].size()-1], 1);
		this->Y_test.row(i) = VectorXd::Map(&strArray[i][strArray[i].size() - 1], 1);
	}
	return true;
}