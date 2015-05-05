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
            const Model* getModel() const {
                return this->model;
            }

            virtual void initialize() = 0;
            virtual void track(const cv::Mat& image) = 0;

        protected:
            cv::Mat templateImage;
            Model* model;
    };
}

#endif //__TRACKER_TRACKER_HPP__
