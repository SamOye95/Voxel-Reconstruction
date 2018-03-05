#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <xtgmath.h>
#include <fstream>


typedef unsigned char uchar;

class ColorModel {
public: 
	ColorModel(int binCount = 8) {
		hist = cv::Mat::zeros(3, 256 / binCount, CV_8UC1);
	}



	void addPoint(uchar r, uchar g, uchar b) {
		hist.at<uchar>(0, floor(r / 255)) += 1;
		hist.at<uchar>(1, floor(g / 255)) += 1;
		hist.at<uchar>(2, floor(b / 255)) += 1;
	}

	int compare(ColorModel *model2) {
		int sumDiff = 0;
		for (int i = 0; i < hist.rows; i++) {
			const uchar* hist_i = hist.ptr<const uchar>(i);
			const uchar* hist2_i = model2->hist.ptr<const uchar>(i);
			for (int j = 0; j < hist.cols; j++)
				sumDiff += abs(hist_i[j] - hist2_i[j]);
		}
		return sumDiff;
	}

	void save(const char * file) {
		std::ofstream outFile(file);
		if (!outFile.is_open()) {
			printf("no");
		};

		for (int i = 0; i < hist.rows; i++)
		{
			const uchar* hist_i = hist.ptr<const uchar>(i);

			for (int j = 0; j < hist.cols; j++)
			{
				outFile << hist_i[j] << ' ';
			}
		}
		outFile << "/n";
		outFile.flush();
		outFile.close();
	}

	void load(const char * file) {
		std::ifstream inFile(file);
		for (int i = 0; i < hist.rows; i++)
		{
			uchar* hist_i = hist.ptr<uchar>(i);

			for (int j = 0; j < hist.cols; j++)
			{
				inFile >> hist_i[j];
			}
		}
		inFile.close();
	}

public:
	cv::Mat hist;
};