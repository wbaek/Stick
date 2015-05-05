#ifndef __TRACKER_INVERSE_COMPOSITIONAL_HPP__
#define __TRACKER_INVERSE_COMPOSITIONAL_HPP__

#include <vector>
#include "tracker/tracker.hpp"

namespace Stick {
    class InverseCompositional : public Tracker {
        public:
            InverseCompositional(Model* model) : Tracker(model) {
            }
            virtual ~InverseCompositional() {
            }

            virtual void initialize();
            virtual void track(const cv::Mat& image);

        protected:
            virtual void calculateGradients(double scale=1.0/255.0);
            virtual void calculateSteepest();
            virtual void calculateHessianInv();

        protected:
            cv::Mat gradients;
            cv::Mat steepest;
            cv::Mat hessianInv;

            cv::Mat transformedImage;
            cv::Mat errorImage;
    };
}

#endif //__TRACKER_INVERSE_COMPOSITIONAL_HPP__
