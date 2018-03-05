#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <xtgmath.h>
#include <fstream>

typedef unsigned char uchar;

class ColorModel {
public:
	ColorModel(int binCount = 8)
	{
		hist = cv::Mat::zeros(3, 256 / binCount, CV_32SC1);
		this->binSize = 256 / binCount;
		this->binCount = binCount;
	}

	void addPoint(int r, int g, int b)
	{
		hist.at<int>(0, floor(r / binCount)) += 1;
		hist.at<int>(1, floor(g / binCount)) += 1;
		hist.at<int>(2, floor(b / binCount)) += 1;
	}

	int compare(ColorModel& model2)
	{
		int sumDiff = 0;
		for (int i = 0; i < hist.rows; i++)
		{
			const int* hist_i = hist.ptr<const int>(i);
			const int* hist2_i = model2.hist.ptr<const int>(i);
			for (int j = 0; j < hist.cols; j++)
			{
				sumDiff += abs(hist_i[j] - hist2_i[j]);
			}
		}
		return sumDiff;
	}

	void save(const char * file)
	{
		std::ofstream outFile(file);
		if (!outFile.is_open())
		{
			printf("File could not be opened.");
		}
		for (int i = 0; i < hist.rows; i++)
		{
			for (int j = 0; j < hist.cols; j++)
			{
				outFile << hist.at<int>(i, j) << " ";
				// eg:
				// outFile << "i :" << i << ", j :" << j << " = " << hist.at<int>(i, j) << " \n"; 
			}
		}
		outFile << "/n";
		outFile.flush();
		outFile.close();
	}

	void load(const char * file)
	{
		std::ifstream inFile(file);
		for (int i = 0; i < hist.rows; i++)
		{
			int* hist_i = hist.ptr<int>(i);
			for (int j = 0; j < hist.cols; j++)
			{
				inFile >> hist_i[j];
			}
		}
		inFile.close();
	}

protected:
	cv::Mat hist;
	int binSize, binCount;
};