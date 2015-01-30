#pragma once

#include <opencv2/opencv.hpp>

namespace Tracking {
    class TemplateTracker {
        protected:
            TemplateTracker() :
                gaussianKernalSize(15),
                maxIteration(100), updateIteration(0), epsilon(1e-3), deltaPose(0.0f), error(0.0f) {
            };
            virtual ~TemplateTracker() {
            }

            cv::Mat templateImage;

            int gaussianKernalSize;

            int maxIteration;
            int updateIteration;
            double epsilon;
            double deltaPose;
            double error;

        public:
            virtual cv::Mat getTemplate() {
                return this->templateImage;
            }
            virtual void setTemplate(const cv::Mat& image) = 0;
            virtual void track(const cv::Mat& image) = 0;

            virtual void setPose(const cv::Mat& pose) = 0;
            virtual cv::Mat getPose() = 0;

            virtual void setGaussianKernalSize(int size) {
                this->gaussianKernalSize = size;
            }

            virtual void setMaxIteration(int iter) {
                this->maxIteration = iter;
            }
            virtual int getMaxIteration() {
                return this->maxIteration;
            }
            virtual int getUpdateIteration() {
                return this->updateIteration;
            }
            virtual void setEpsilon(double epsilon=0.05) {
                this->epsilon = epsilon;
            }
            virtual double getEpsilon() {
                return this->epsilon;
            }
            virtual double getDeltaPose() {
                return this->deltaPose;
            }
            virtual double getError() {
                return this->error;
            }
    };
}
