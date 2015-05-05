#ifndef __TRACKER_INVERSE_COMPOSITIONAL_HPP__
#define __TRACKER_INVERSE_COMPOSITIONAL_HPP__

#include <vector>
#include <utils/string.hpp>

#include "tracker/tracker.hpp"

namespace Stick {
    class InverseCompositional : public Tracker {
        public:
            InverseCompositional(Model* model, double thresholdSumOfComposeDelta=0.5, int maxIteration=100) : Tracker(model) {
                this->thresholdSumOfComposeDelta = thresholdSumOfComposeDelta;
                this->maxIteration = maxIteration;
            }
            virtual ~InverseCompositional() {
            }

            virtual void initialize();
            virtual void track(const cv::Mat& image, const double scale=1.0);
            virtual std::vector<cv::Mat> getPoseTrace() const {
                return this->poseTrace;
            }
            virtual std::string getLogString() const {
                return instant::Utils::String::Format("iter:%d, delta:%.2f",
                        this->iter, this->sumOfComposeDelta);
            }

            virtual cv::Mat getErrorImage() const {
                return this->errorImage.clone();
            }

        protected:
            virtual void calculateGradients(double scale=1.0);
            virtual void calculateSteepest();
            virtual void calculateHessianInv();

        protected:
            cv::Mat gradients;
            cv::Mat steepest;
            cv::Mat hessianInv;

            cv::Mat errorImage;

            double sumOfComposeDelta;
            int iter;

            double thresholdSumOfComposeDelta;
            int maxIteration;
            std::vector<cv::Mat> poseTrace;
    };
}

#endif //__TRACKER_INVERSE_COMPOSITIONAL_HPP__
