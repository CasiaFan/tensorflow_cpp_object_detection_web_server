#ifndef _detect
#define _detect

#include <vector>
#include "tensorflow/core/public/session.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

using namespace cv;

class Detector {
    std::unique_ptr<tensorflow::Session> session;
    public:
        int loadModel(std::string modelPath);
        int detect(Mat frame, double thresholdScore, double thresholdIOU, std::vector<float> &boxes, std::vector<float> &scores, std::vector<size_t> &labels);
};
#endif

