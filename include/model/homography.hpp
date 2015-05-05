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
            virtual void compose(const cv::Mat& delta) {
                this->pose = this->pose * delta;
                this->pose /= this->pose.at<double>(8);
            }
            virtual cv::Mat inverse() const {
                cv::Mat inv = this->pose.inv();
                return inv / inv.at<double>(8);
            }
    };
}


#endif //__MODEL_HOMOGRAPHY_HPP__
