#ifndef __MODEL_HOMOGRAPHY_HPP__
#define __MODEL_HOMOGRAPHY_HPP__

#include "model/model.hpp"

namespace Stick {
    class Homography : public Model {
        public:
            Homography() {
                this->pose = cv::Mat::eye(cv::Size(3, 3), cv::DataType<double>::type);
            }
            virtual ~Homography() {
            }

        public:
            virtual int getParameterSize() const {
                return 8;
            }

            virtual void compose(const cv::Mat& delta) {
                this->pose = this->pose * delta;
                this->pose /= this->pose.at<double>(8);
            }
            virtual cv::Mat inverse() const {
                cv::Mat inv = this->pose.inv();
                return inv / inv.at<double>(8);
            }

            virtual cv::Mat jacobian(const cv::Point& at) const {
                cv::Mat in = cv::Mat::ones(cv::Size(1, 3), cv::DataType<double>::type);
                in.at<double>(0) = at.x;
                in.at<double>(1) = at.y;
                in.at<double>(2) = 1.0;
                return this->jacobian(in);
            }
            virtual cv::Mat jacobian(const cv::Mat& in) const {
                cv::Mat prime = this->pose * in;
                double z = prime.at<double>(2);

                cv::Mat out = cv::Mat::zeros(cv::Size(9, 2), cv::DataType<double>::type);
                out.at<double>(0, 0) = in.at<double>(0) / z;
                out.at<double>(0, 1) = in.at<double>(1) / z;
                out.at<double>(0, 2) = in.at<double>(2) / z;
                out.at<double>(0, 6) = - in.at<double>(0) * prime.at<double>(0);
                out.at<double>(0, 7) = - in.at<double>(1) * prime.at<double>(0);

                out.at<double>(1, 3) = in.at<double>(0) / z;
                out.at<double>(1, 4) = in.at<double>(1) / z;
                out.at<double>(1, 5) = in.at<double>(2) / z;
                out.at<double>(1, 6) = - in.at<double>(0) * prime.at<double>(1);
                out.at<double>(1, 7) = - in.at<double>(1) * prime.at<double>(1);

                return out;
            }
    };
}


#endif //__MODEL_HOMOGRAPHY_HPP__
