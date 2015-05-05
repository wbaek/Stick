#ifndef __MODEL_MODEL_HPP__
#define __MODEL_MODEL_HPP__

#include <opencv2/opencv.hpp>
#include <utils/string.hpp>
#include <utils/type.hpp>

#include "exceptions/exception.hpp"
#include "exceptions/invalid_parameters.hpp"

namespace Stick {
    class Model {
        protected:
            Model() {
            }
        public:
            virtual ~Model() {
            }
            virtual std::string getName() const {
                std::string className = instant::Utils::Type::GetTypeName(this);
                return instant::Utils::String::Replace(className, "Stick::", "");
            }

            virtual void set(const cv::Mat& pose) {
                if( this->pose.size() != pose.size() ) {
                    std::string message = instant::Utils::String::Format(
                        "not matched pose size (input:%dx%d, target:%dx%d)",
                        pose.size().width, pose.size().height,
                        this->pose.size().width, this->pose.size().height);

                    MakeClassException(InvalidParameters, message);
                }
                pose.copyTo(this->pose);
            }
            virtual const cv::Mat get() const {
                return this->pose.clone();
            }

            virtual void initialize() {
                this->pose = cv::Mat::eye(this->pose.size(), this->pose.type());
            }
            virtual cv::Point transform(const cv::Point& point) const {
                cv::Mat in = cv::Mat::ones(cv::Size(1, 3), cv::DataType<double>::type);
                in.at<double>(0) = point.x;
                in.at<double>(1) = point.y;
                in.at<double>(2) = 1.0;

                cv::Mat out = this->transform(in);
                return cv::Point(out.at<double>(0)+0.5, out.at<double>(1)+0.5);
            }
            virtual cv::Mat transform(const cv::Mat& point) const {
                cv::Mat in(point);
                in.at<double>(2) = 1.0;
                cv::Mat out = this->pose * in;

                double z = 1.0;
                if( out.size().height == 3 ) {
                    z = out.at<double>(2);
                }
                out = out/z;
                return out;
            }

            virtual int getParameterSize() const = 0;

            virtual void compose(const cv::Mat& delta) = 0;
            virtual cv::Mat inverse() const = 0;

            virtual cv::Mat jacobian(const cv::Point& at) const =0;
            virtual cv::Mat jacobian(const cv::Mat& in) const = 0;

        protected:
            cv::Mat pose;
    };
}

#endif //__MODEL_MODEL_HPP__
