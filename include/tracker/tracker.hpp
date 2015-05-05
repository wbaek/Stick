#ifndef __TRACKER_TRACKER_HPP__
#define __TRACKER_TRACKER_HPP__

#include <opencv2/opencv.hpp>
#include <utils/string.hpp>
#include <utils/type.hpp>

#include "model/model.hpp"
#include "exceptions/invalid_parameters.hpp"

namespace Stick {
    class Tracker {
        protected:
            Tracker(Model* model) {
                this->model = model;
            }

        public:
            virtual ~Tracker() {
                if(this->model) delete this->model;
                this->model = NULL;
            }
            virtual std::string getName() const {
                std::string className = instant::Utils::Type::GetTypeName(this);
                return instant::Utils::String::Replace(className, "Stick::", "");
            }

            void setTemplate(const cv::Mat& image) {
                if(image.channels() != 1) {
                    throw MakeClassException(InvalidParameters, "template image must be a single channel");
                }
                image.copyTo( this->templateImage );
            }
            const cv::Mat getTemplate() const {
                return this->templateImage.clone();
            }
            const cv::Mat getTransformedImage(const cv::Mat& image, const cv::Size& templateSize) {
                int dx = image.size().width/2 - templateSize.width/2;
                int dy = image.size().height/2 - templateSize.height/2;

                cv::Mat pose = this->model->get();
                pose.at<double>(cv::Point(2, 0)) += dx;
                pose.at<double>(cv::Point(2, 1)) += dy;

                cv::warpPerspective(image, this->transformedImage, pose.inv(), templateSize);
                return this->transformedImage.clone();
            }

            const Model* getModel() const {
                return this->model;
            }

            virtual void initialize() = 0;
            virtual void track(const cv::Mat& image) = 0;
            virtual std::vector<cv::Mat> getPoseTrace() const = 0;
            virtual std::string getLogString() const = 0;

        protected:
            cv::Mat templateImage;
            cv::Mat transformedImage;
            Model* model;
    };
}

#endif //__TRACKER_TRACKER_HPP__
