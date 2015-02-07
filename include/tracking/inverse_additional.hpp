#pragma once

#include <vector>
#include "template_tracker.hpp"

namespace Tracking {
    class InverseAdditional : public TemplateTracker{
        protected:
            cv::Mat model;
            cv::Mat descent;
            cv::Mat hessian;
            cv::Mat updater;
            std::vector<cv::Mat> updatePoseList;

        public:
            InverseAdditional() : TemplateTracker() {
                this->model = cv::Mat::eye(3, 3, cv::DataType<double>::type);
            };
            virtual ~InverseAdditional() {
            }
            
            virtual void setPose(const cv::Mat& pose) {
                this->model = pose.clone();
            }
            virtual cv::Mat getPose() {
                return this->model.clone();
            }
            virtual std::vector<cv::Mat> getUpdatePoseList() {
                return this->updatePoseList;
            }

            cv::Point transform(const cv::Point& point) {
                cv::Mat in(3, 1, cv::DataType<double>::type);
                in.at<double>(0) = point.x;
                in.at<double>(1) = point.y;
                in.at<double>(2) = 1.0;
                cv::Mat out = this->model * in;

                double z = out.at<double>(2);
                return cv::Point(out.at<double>(0)/z, out.at<double>(1)/z);
            }

            virtual void setTemplate(const cv::Mat& image, const cv::Size& size) {
                cv::Mat blurred = image.clone();
                cv::GaussianBlur(blurred, blurred,
                        cv::Size(this->gaussianKernalSize, this->gaussianKernalSize), 0);

                this->templateImage = this->getTrackedImage(blurred, size);
                std::vector<cv::Mat> gradient = this->calcGradient(this->templateImage);
                this->descent = this->calcSteepestDescent( gradient );

                // calc hessian
                this->hessian = (this->descent * this->descent.t()).inv();
                this->updater = this->hessian * this->descent;
            }

            virtual void track(const cv::Mat& image) {
                this->updatePoseList.clear();

                cv::Mat blurred(image.size(), cv::DataType<unsigned char>::type);
                cv::GaussianBlur(image, blurred,
                        cv::Size(this->gaussianKernalSize, this->gaussianKernalSize), 0);
                cv::imshow("blurred", blurred);

                int iter=0;
                for(iter=0; iter<this->maxIteration; iter++) {
                    cv::Mat errorImage = this->calcEror( blurred );
                    cv::Mat errorVector = errorImage.reshape(0, errorImage.size().area());
                    cv::Mat deltaModel = this->updater * errorVector;
                    deltaModel = deltaModel.reshape(0, 3);

                    this->model = this->model - deltaModel;
                    this->updatePoseList.push_back(this->model.clone());

                    this->deltaPose = cv::norm(deltaModel, cv::NORM_L1);
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
                        cv::Point imagePoint = this->transform(pt) + delta;
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
                for(int y=1; y<image.size().height-1; y++) {
                    for(int x=1; x<image.size().width-1; x++) {
                        gradient[0].at<double>(y,x) = (double)image.at<unsigned char>(y,x+1) - (double)image.at<unsigned char>(y,x-1);
                        gradient[1].at<double>(y,x) = (double)image.at<unsigned char>(y+1,x) - (double)image.at<unsigned char>(y-1,x);
                    }
                }

                return gradient;
            }

            cv::Mat calcSteepestDescent(std::vector<cv::Mat>& gradient) {
                cv::Size size = gradient[0].size();
                cv::Point delta(size.width/2, size.height/2);
                cv::Mat descent = cv::Mat::zeros(9, size.area(), cv::DataType<double>::type);
                for(int y=0; y<size.height; y++) {
                    for(int x=0; x<size.width; x++) {
                        descent.at<double>(0, y*size.width+x) = ((double)x) * gradient[0].at<double>(y, x);
                        descent.at<double>(1, y*size.width+x) = ((double)y) * gradient[0].at<double>(y, x);
                        descent.at<double>(2, y*size.width+x) = ((double)1) * gradient[0].at<double>(y, x);
                        descent.at<double>(3, y*size.width+x) = ((double)x) * gradient[1].at<double>(y, x);
                        descent.at<double>(4, y*size.width+x) = ((double)y) * gradient[1].at<double>(y, x);
                        descent.at<double>(5, y*size.width+x) = ((double)1) * gradient[1].at<double>(y, x);

                        descent.at<double>(6, y*size.width+x) = ((double)-x*x) * gradient[0].at<double>(y, x)
                                                              + ((double)-x*y) * gradient[1].at<double>(y, x);
                        descent.at<double>(7, y*size.width+x) = ((double)-x*y) * gradient[0].at<double>(y, x)
                                                              + ((double)-y*y) * gradient[1].at<double>(y, x);
                        descent.at<double>(8, y*size.width+x) = ((double)-x)   * gradient[0].at<double>(y, x)
                                                              + ((double)-y)   * gradient[1].at<double>(y, x);
                    }
                }

                return descent.clone();
            }

            cv::Mat calcEror(const cv::Mat& image) {
                cv::Point delta(image.size().width/2, image.size().height/2);
                cv::Point templateDelta(this->templateImage.size().width/2, this->templateImage.size().height/2);

                cv::Mat error = cv::Mat::zeros(this->templateImage.size(), cv::DataType<double>::type);
                for(int y=-templateDelta.y; y<templateDelta.y; y++) {
                    for(int x=-templateDelta.x; x<templateDelta.x; x++) {
                        cv::Point pt(x, y);
                        cv::Point imagePoint = this->transform(pt) + delta;
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
