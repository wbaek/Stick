#include "tracker/inverse_compositional.hpp"

#include "exceptions/not_initialized.hpp"

using namespace Stick;

void InverseCompositional::initialize() {
    this->calculateGradients();
    this->calculateSteepest();
    this->calculateHessianInv();

    this->errorImage = cv::Mat::zeros(this->templateImage.size(), cv::DataType<double>::type);
}

void InverseCompositional::track(const cv::Mat& image, const double scale) {
    int width = this->templateImage.size().width;
    int height = this->templateImage.size().height;
    int params = this->model->getParameterSize();

    this->poseTrace.clear();
    for(int i=0; i<this->maxIteration; i++) {
        this->calculateTransformedImage(image, this->templateImage.size());
        for(int y=0; y<height; y++) {
            for(int x=0; x<width; x++) {
                cv::Point pt(x, y);
                this->errorImage.at<double>(pt)
                    = ((double)this->transformedImage.at<unsigned char>(pt)
                    - (double)this->templateImage.at<unsigned char>(pt)) * scale;
            }
        }
        cv::Mat reshapedError = this->errorImage.reshape(0, width*height);

        cv::Mat pose = this->model->get();
        cv::Mat delta = this->hessianInv * (this->steepest * reshapedError);
        cv::Mat deltaPose = cv::Mat::eye(pose.size(), cv::DataType<double>::type);
        for(int i=0; i<delta.size().area(); i++) {
            deltaPose.at<double>(i) += delta.at<double>(i);
        }

        this->model->set(deltaPose);
        cv::Mat deltaInv = this->model->inverse();

        this->model->set(pose);
        this->model->compose(deltaInv);
        this->poseTrace.push_back(this->model->get());

        double sumOfComposeDelta = -2.0;
        for(int p=0; p<params; p++) {
            sumOfComposeDelta += std::abs(deltaInv.at<double>(p));
        }
       
        this->iter = i;
        this->sumOfComposeDelta = sumOfComposeDelta;
        if( sumOfComposeDelta < this->thresholdSumOfComposeDelta ) {
            break;
        }
    }
}

void InverseCompositional::calculateGradients(double scale){
    if(this->templateImage.size().area() == 0) {
        throw MakeClassException(NotInitialized, "template image not initialized");
    }

    cv::Mat image = this->templateImage;
    this->gradients = cv::Mat::zeros(cv::Size(image.size().area(), 2), cv::DataType<double>::type);

    int width = image.size().width;
    int height = image.size().height;
    for(int y=1; y<height-1; y++) {
        for(int x=1; x<width-1; x++) {
            this->gradients.at<double>(cv::Point(y*width+x, 0)) = ((double)image.at<unsigned char>(y,x+1) - (double)image.at<unsigned char>(y,x-1)) * scale;
            this->gradients.at<double>(cv::Point(y*width+x, 1)) = ((double)image.at<unsigned char>(y+1,x) - (double)image.at<unsigned char>(y-1,x)) * scale;
        }
    }
}

void InverseCompositional::calculateSteepest() {
    int width = this->templateImage.size().width;
    int height = this->templateImage.size().height;
    int params = this->model->getParameterSize();
    this->steepest = cv::Mat::zeros(cv::Size(width*height, params), cv::DataType<double>::type);

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            cv::Mat in = cv::Mat::ones(cv::Size(1, 3), cv::DataType<double>::type);
            in.at<double>(0) = (double)x - (double)width/2.0;
            in.at<double>(1) = (double)y - (double)height/2.0;
            in.at<double>(2) = 1.0;
            
            cv::Mat out = this->model->jacobian(in);
            for(int p=0; p<this->model->getParameterSize(); p++) {
                this->steepest.at<double>(cv::Point(y*width+x, p))
                    = out.at<double>(cv::Point(p, 0)) * this->gradients.at<double>(cv::Point(y*width+x, 0))
                    + out.at<double>(cv::Point(p, 1)) * this->gradients.at<double>(cv::Point(y*width+x, 1));
            }

        }
    }
}

void InverseCompositional::calculateHessianInv() {
    cv::Mat temp = (this->steepest * this->steepest.t());
    this->hessianInv = temp.inv();
}

