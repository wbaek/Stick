#pragma once

#include <opencv2/opencv.hpp>

namespace Tracking {
    class Model {
        public:
            virtual void setModel(const cv::Mat& model) = 0;
            virtual cv::Mat getModel() = 0;
            virtual cv::Point transform(const cv::Point& point) = 0;
            virtual cv::Mat inverse() = 0;
            virtual cv::Mat compose(const cv::Mat& delta) = 0;
    };
}
