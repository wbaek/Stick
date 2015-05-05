#include <gtest/gtest.h>

#include <tracker/inverse_compositional.hpp>
#include <model/homography.hpp>

namespace Stick {
    class InverseCompositionalTest : public InverseCompositional {
        public:
            InverseCompositionalTest(Model* model) : InverseCompositional(model) {
            }

            void calculateGradients() {
                InverseCompositional::calculateGradients();
            }
            cv::Mat getGradients() const {
                return this->gradients.clone();
            }

            void calculateSteepest() {
                InverseCompositional::calculateSteepest();
            }
            cv::Mat getSteepest() const {
                return this->steepest.clone();
            }

            void calculateHessianInv() {
                InverseCompositional::calculateHessianInv();
            }
            cv::Mat getHessianInv() const {
                return this->hessianInv.clone();
            }

            cv::Mat getTransformedImage() const {
                return this->transformedImage.clone();
            }
            cv::Mat getErrorImage() const {
                return this->errorImage.clone();
            }
    };
}

TEST(InverseCompositional, create) {
    Stick::InverseCompositional tracker(new Stick::Homography());
}

TEST(InverseCompositional, calculate_gradient) {
    Stick::InverseCompositionalTest tracker(new Stick::Homography());

    cv::Mat image = cv::imread("datas/im000.png", CV_LOAD_IMAGE_GRAYSCALE);

    tracker.setTemplate( image );
    tracker.calculateGradients();

    cv::Mat gradients = tracker.getGradients();
    gradients = (gradients/2.0 + 0.5) * 255.0;
    cv::Mat gradientImage = gradients.reshape(1, image.size().height*2);

    cv::imwrite("gradient.png", gradientImage);
}

TEST(InverseCompositional, calculate_steepest) {
    Stick::InverseCompositionalTest tracker(new Stick::Homography());

    cv::Mat image = cv::imread("datas/im000.png", CV_LOAD_IMAGE_GRAYSCALE);

    tracker.setTemplate( image );
    tracker.calculateGradients();
    tracker.calculateSteepest();

    cv::Mat steepest = tracker.getSteepest();
    steepest = (steepest/2.0 + 0.5) * 255.0;
    cv::Mat steepestImage = steepest.reshape(1, image.size().height*8);

    cv::imwrite("steepest.png", steepestImage);
}

TEST(InverseCompositional, calculate_hessian_inv) {
    Stick::InverseCompositionalTest tracker(new Stick::Homography());

    cv::Mat image = cv::imread("datas/im000.png", CV_LOAD_IMAGE_GRAYSCALE);

    tracker.setTemplate( image );
    tracker.calculateGradients();
    tracker.calculateSteepest();
    tracker.calculateHessianInv();

    cv::Mat hessianInv = tracker.getHessianInv();
    std::cout << hessianInv << std::endl;
}

TEST(InverseCompositional, calculate_track) {
    Stick::InverseCompositionalTest tracker(new Stick::Homography());

    cv::Mat templateImage = cv::imread("datas/im000.png", CV_LOAD_IMAGE_GRAYSCALE);

    tracker.setTemplate( templateImage );
    tracker.initialize();

    cv::Mat image = cv::imread("datas/im001_original.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    tracker.track( image );

    cv::imwrite("transformed.png", tracker.getTransformedImage());
    cv::imwrite("error.png", (tracker.getErrorImage() + 128));

}


