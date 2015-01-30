#pragma once

#include <opencv2/opencv.hpp>

namespace Tracking {
    class Affine {
        private:
            cv::Mat model;

        public:
            Affine() {
                this->model = cv::Mat::zeros(3, 2, cv::DataType<double>::type);
            }
            virtual ~Affine() {
            }

            void setModel(const cv::Mat& model) {
                this->model = model.clone();
            }
            cv::Mat getModel() {
                return this->model;
            }

            cv::Point transform(const cv::Point& point) {
                double dx = (1.0+this->model.at<double>(0))*point.x +      this->model.at<double>(2) *point.y + this->model.at<double>(4);
                double dy =      this->model.at<double>(1) *point.x + (1.0+this->model.at<double>(3))*point.y + this->model.at<double>(5);
                return cv::Point(dx, dy);
            }

            cv::Mat inverse() {
                const cv::Mat p = this->model;
                cv::Mat modelInv(3, 2, cv::DataType<double>::type);

                double factor = (1.0f+p.at<double>(0))*(1.0f+p.at<double>(3))-(p.at<double>(1)*p.at<double>(2));
                modelInv.at<double>(0) = (-p.at<double>(0)-p.at<double>(0)*p.at<double>(3)+p.at<double>(1)*p.at<double>(2));
                modelInv.at<double>(1) = (-p.at<double>(1));
                modelInv.at<double>(2) = (-p.at<double>(2));
                modelInv.at<double>(3) = (-p.at<double>(3)-p.at<double>(0)*p.at<double>(3)+p.at<double>(1)*p.at<double>(2));
                modelInv.at<double>(4) = (-p.at<double>(4)-p.at<double>(3)*p.at<double>(4)+p.at<double>(2)*p.at<double>(5));
                modelInv.at<double>(5) = (-p.at<double>(5)-p.at<double>(0)*p.at<double>(5)+p.at<double>(1)*p.at<double>(4));

                return modelInv / factor;
            }
            cv::Mat compose(const cv::Mat& delta) {
                cv::Mat updated(3, 2, cv::DataType<double>::type);

                updated.at<double>(0) = this->model.at<double>(0) + delta.at<double>(0)
                    + this->model.at<double>(0) * delta.at<double>(0) + this->model.at<double>(2) * delta.at<double>(1);
                updated.at<double>(1) = this->model.at<double>(1) + delta.at<double>(1)
                    + this->model.at<double>(1) * delta.at<double>(0) + this->model.at<double>(3) * delta.at<double>(1);
                updated.at<double>(2) = this->model.at<double>(2) + delta.at<double>(2)
                    + this->model.at<double>(0) * delta.at<double>(2) + this->model.at<double>(2) * delta.at<double>(3);
                updated.at<double>(3) = this->model.at<double>(3) + delta.at<double>(3)
                    + this->model.at<double>(1) * delta.at<double>(2) + this->model.at<double>(3) * delta.at<double>(3);
                updated.at<double>(4) = this->model.at<double>(4) + delta.at<double>(4)
                    + this->model.at<double>(0) * delta.at<double>(4) + this->model.at<double>(2) * delta.at<double>(5);
                updated.at<double>(5) = this->model.at<double>(5) + delta.at<double>(5)
                    + this->model.at<double>(1) * delta.at<double>(4) + this->model.at<double>(3) * delta.at<double>(5);

                return updated;
            }
    };
}
