#pragma once

#include <vector>
#include "template_tracker.hpp"

namespace Tracking {
    template <typename Model>
    class InverseCompositional : public TemplateTracker{
        protected:
            Model model;
            cv::Mat descent;
            cv::Mat hessian;
            std::vector<cv::Mat> updatePoseList;

        public:
            InverseCompositional() : TemplateTracker() {
            };
            virtual ~InverseCompositional() {
            }
            
            virtual void setPose(const cv::Mat& pose) {
                this->model.setModel( pose );
            }
            virtual cv::Mat getPose() {
                return this->model.getModel();
            }
            virtual std::vector<cv::Mat> getUpdatePoseList() {
                return this->updatePoseList;
            }

            virtual void setTemplate(const cv::Mat& image) {
                this->templateImage = image.clone();
                cv::GaussianBlur(this->templateImage, this->templateImage,
                        cv::Size(this->gaussianKernalSize, this->gaussianKernalSize), 0);
                std::vector<cv::Mat> gradient = this->calcGradient(this->templateImage);
                this->calcSteepestDescent( gradient );

                // calc hessian
                this->hessian = (this->descent * this->descent.t()).inv();
            }

            virtual void track(const cv::Mat& image) {
                this->updatePoseList.clear();

                cv::Mat blurred(image.size(), cv::DataType<unsigned char>::type);
                cv::GaussianBlur(image, blurred,
                        cv::Size(this->gaussianKernalSize, this->gaussianKernalSize), 0);

                int iter=0;
                for(iter=0; iter<this->maxIteration; iter++) {
                    cv::Mat errorImage = this->calcEror( blurred );
                    cv::Mat errorVector = errorImage.reshape(0, errorImage.size().area());
                    cv::Mat deltaPose = this->hessian * (this->descent * errorVector);

                    Model invDeltaPose; invDeltaPose.setModel(deltaPose);
                    cv::Mat affine = this->model.compose(invDeltaPose.inverse());

                    this->model.setModel(affine);
                    this->updatePoseList.push_back(affine);

                    this->deltaPose = cv::norm(deltaPose, cv::NORM_L1);
                    this->error = cv::norm(errorVector, cv::NORM_L1) / (double)(errorVector.size().area()*255);
                    if(this->deltaPose < this->epsilon)
                        break;
                }
                this->updateIteration = iter;
            }

            virtual cv::Mat getTrackedImage(const cv::Mat& image, const cv::Size& size) {
                cv::Point delta(image.size().width/2, image.size().height/2);
                cv::Point templateDelta(size.width/2, size.height/2);

                cv::Mat posedImage = cv::Mat::zeros(size, cv::DataType<unsigned char>::type);
                for(int y=-templateDelta.y; y<templateDelta.y; y++) {
                    for(int x=-templateDelta.x; x<templateDelta.x; x++) {
                        cv::Point pt(x, y);
                        cv::Point imagePoint = this->model.transform(pt) + delta;
                        if( 0 <= imagePoint.x && imagePoint.x < image.size().width &&
                            0 <= imagePoint.y && imagePoint.y < image.size().height ) {
                            posedImage.at<unsigned char>(pt + templateDelta)
                                = image.at<unsigned char>(imagePoint);
                        }
                    }
                }
                return posedImage;
            }

        protected:
            std::vector<cv::Mat> calcGradient(const cv::Mat& image) {
                std::vector<cv::Mat> gradient(2);
                gradient[0].create(image.size(), cv::DataType<double>::type);
                gradient[1].create(image.size(), cv::DataType<double>::type);
                for(int y=0; y<image.size().height-1; y++) {
                    for(int x=0; x<image.size().width-1; x++) {
                        gradient[0].at<double>(y,x) = (double)image.at<unsigned char>(y,x+1) - (double)image.at<unsigned char>(y,x);
                        gradient[1].at<double>(y,x) = (double)image.at<unsigned char>(y+1,x) - (double)image.at<unsigned char>(y,x);
                    }
                }
                return gradient;
            }

            void calcSteepestDescent(std::vector<cv::Mat>& gradient) {
                cv::Size size = gradient[0].size();
                cv::Point delta(size.width/2, size.height/2);
                cv::Mat descent = cv::Mat::zeros(6, size.area(), cv::DataType<double>::type);
                for(int y=0; y<size.height; y++) {
                    for(int x=0; x<size.width; x++) {
                        descent.at<double>(0, y*size.width+x) = ((double)x) * gradient[0].at<double>(y, x);
                        descent.at<double>(1, y*size.width+x) = ((double)x) * gradient[1].at<double>(y, x);
                        descent.at<double>(2, y*size.width+x) = ((double)y) * gradient[0].at<double>(y, x);
                        descent.at<double>(3, y*size.width+x) = ((double)y) * gradient[1].at<double>(y, x);
                        descent.at<double>(4, y*size.width+x) = gradient[0].at<double>(y, x);
                        descent.at<double>(5, y*size.width+x) = gradient[1].at<double>(y, x);
                    }
                }

                this->descent = descent.clone();
            }

            cv::Mat calcEror(const cv::Mat& image) {
                cv::Point delta(image.size().width/2, image.size().height/2);
                cv::Point templateDelta(this->templateImage.size().width/2, this->templateImage.size().height/2);

                cv::Mat error = cv::Mat::zeros(this->templateImage.size(), cv::DataType<double>::type);
                for(int y=-templateDelta.y; y<templateDelta.y; y++) {
                    for(int x=-templateDelta.x; x<templateDelta.x; x++) {
                        cv::Point pt(x, y);
                        cv::Point imagePoint = this->model.transform(pt) + delta;
                        if( 0 <= imagePoint.x && imagePoint.x < image.size().width &&
                            0 <= imagePoint.y && imagePoint.y < image.size().height ) {
                            error.at<double>(pt + templateDelta) =
                                (double)image.at<unsigned char>(imagePoint) 
                                - (double)templateImage.at<unsigned char>(pt + templateDelta);
                        }
                    }
                }
                return error;
            }
    };
}
