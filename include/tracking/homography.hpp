#pragma once
#include "model.hpp"
#include <opencv2/opencv.hpp>

namespace Tracking {
    class Homography : public Model {
        private:
            cv::Mat model;

        public:
            Homography() {
                this->model = cv::Mat::eye(3, 3, cv::DataType<double>::type);
            }
            virtual ~Homography() {
            }

            void setModel(const cv::Mat& model) {
                this->model = model.clone();
            }
            cv::Mat getModel() {
                return this->model;
            }
            static cv::Mat getInitial() {
                return cv::Mat::eye(3, 3, cv::DataType<double>::type);
            }

            cv::Point transform(const cv::Point& point) {
                cv::Mat in = cv::Mat::ones(3, 1, cv::DataType<double>::type);
                in.at<double>(0) = point.x;
                in.at<double>(1) = point.y;
                cv::Mat out = this->model * in;

                double z = out.at<double>(2);
                return cv::Point(out.at<double>(0)/z, out.at<double>(1)/z);
            }

            cv::Mat inverse() {
                const cv::Mat p = this->model;
                return p.inv();
            }
            cv::Mat compose(const cv::Mat& delta) {
                return this->model * delta;
            }
    };
}
