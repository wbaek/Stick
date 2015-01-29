#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

#include <utils/string.hpp>
#include <opencv2/opencv.hpp>

void help(char* execute) {
    std::cerr << "usage: " << execute << " [-h] -p DATA_PATH" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "\t-h, --help              show this help message and exit" << std::endl;
    std::cerr << "\t-p, --path DATA_PATH    set DATA_PATH" << std::endl;
    exit(-1);
}

cv::Point transform(const cv::Mat& affine, float x, float y) {
    float dx = (1.0+affine.at<float>(0))*x +      affine.at<float>(2) *y + affine.at<float>(4);
    float dy =      affine.at<float>(1) *x + (1.0+affine.at<float>(3))*y + affine.at<float>(5);
    return cv::Point(dx, dy);
}

cv::Mat inverseAffine(const cv::Mat& p) {
    cv::Mat inverse(3, 2, cv::DataType<float>::type);

    float factor = ((1.0f+p.at<float>(0))*(1.0f+p.at<float>(3))-(p.at<float>(1)*p.at<float>(2)));
    inverse.at<float>(0) = (-p.at<float>(0)-p.at<float>(0)*p.at<float>(3)+p.at<float>(1)*p.at<float>(2));
    inverse.at<float>(1) = (-p.at<float>(1));
    inverse.at<float>(2) = (-p.at<float>(2));
    inverse.at<float>(3) = (-p.at<float>(3)-p.at<float>(0)*p.at<float>(3)+p.at<float>(1)*p.at<float>(2));
    inverse.at<float>(4) = (-p.at<float>(4)-p.at<float>(3)*p.at<float>(4)+p.at<float>(2)*p.at<float>(5));
    inverse.at<float>(5) = (-p.at<float>(5)-p.at<float>(0)*p.at<float>(5)+p.at<float>(1)*p.at<float>(4));

    return inverse / factor;
}

cv::Mat updateAffineCompositional(const cv::Mat& affine, const cv::Mat& delta) {
    cv::Mat updated(3, 2, cv::DataType<float>::type);

    updated.at<float>(0) = affine.at<float>(0) + delta.at<float>(0)
        + affine.at<float>(0) + delta.at<float>(0) + affine.at<float>(2) + delta.at<float>(1);
    updated.at<float>(1) = affine.at<float>(1) + delta.at<float>(1)
        + affine.at<float>(1) + delta.at<float>(0) + affine.at<float>(3) + delta.at<float>(1);
    updated.at<float>(2) = affine.at<float>(2) + delta.at<float>(2)
        + affine.at<float>(0) + delta.at<float>(2) + affine.at<float>(2) + delta.at<float>(3);
    updated.at<float>(3) = affine.at<float>(3) + delta.at<float>(3)
        + affine.at<float>(1) + delta.at<float>(2) + affine.at<float>(3) + delta.at<float>(3);
    updated.at<float>(4) = affine.at<float>(4) + delta.at<float>(4)
        + affine.at<float>(0) + delta.at<float>(4) + affine.at<float>(2) + delta.at<float>(5);
    updated.at<float>(5) = affine.at<float>(5) + delta.at<float>(5)
        + affine.at<float>(1) + delta.at<float>(4) + affine.at<float>(3) + delta.at<float>(5);

    return updated;
}

std::vector<cv::Mat> calcGradient(const cv::Mat& image) {
    std::vector<cv::Mat> gradient(2);
    gradient[0].create(image.size(), cv::DataType<float>::type);
    gradient[1].create(image.size(), cv::DataType<float>::type);
    for(int y=0; y<image.size().height-1; y++) {
        for(int x=0; x<image.size().width-1; x++) {
            gradient[0].at<float>(y, x) = ((float)image.at<unsigned char>(y, x+1) - (float)image.at<unsigned char>(y, x));
            gradient[1].at<float>(y, x) = ((float)image.at<unsigned char>(y+1, x) - (float)image.at<unsigned char>(y, x));
        }
    }
    return gradient;
}

cv::Mat calcSteepestDescent(std::vector<cv::Mat>& gradient) {
    cv::Size size = gradient[0].size();
    cv::Point delta(size.width/2, size.height/2);
    cv::Mat descent(6, size.area(), cv::DataType<float>::type);
    for(int y=0; y<size.height; y++) {
        for(int x=0; x<size.width; x++) {
            descent.at<float>(0, y*size.width+x) = (float)(x) * gradient[0].at<float>(y, x);
            descent.at<float>(1, y*size.width+x) = (float)(x) * gradient[1].at<float>(y, x);
            descent.at<float>(2, y*size.width+x) = (float)(y) * gradient[0].at<float>(y, x);
            descent.at<float>(3, y*size.width+x) = (float)(y) * gradient[1].at<float>(y, x);
            descent.at<float>(4, y*size.width+x) = gradient[0].at<float>(y, x);
            descent.at<float>(5, y*size.width+x) = gradient[1].at<float>(y, x);
        }
    }

    return descent;
}

cv::Mat calcError(const cv::Mat& image, const cv::Mat& templateImage, const cv::Mat& affine) {
    cv::Point delta(image.size().width/2, image.size().height/2);
    cv::Point templateDelta(templateImage.size().width/2, templateImage.size().height/2);

    cv::Mat error(templateImage.size(), cv::DataType<float>::type);
    for(int y=-templateImage.size().height/2; y<templateImage.size().height/2; y++) {
        for(int x=-templateImage.size().width/2; x<templateImage.size().width/2; x++) {
            error.at<float>(cv::Point(x, y) + templateDelta)
                = ( (float)image.at<unsigned char>(transform(affine, x, y) + delta)
                        - (float)templateImage.at<unsigned char>(cv::Point(x, y) + templateDelta) );
        }
    }
    return error;
}

void copyTo(cv::Mat& image, cv::Mat& templateImage, cv::Mat& affineInv) {
    cv::Size sourceSize = image.size();
    cv::Size targetSize = templateImage.size();

    for(int y = -targetSize.height/2; y < targetSize.height/2; y++) {
        for(int x = -targetSize.width/2; x < targetSize.width/2; x++) {
            cv::Point d = transform(affineInv, x, y);
            templateImage.at<unsigned char>(y + targetSize.height/2, x + targetSize.width/2)
                = image.at<unsigned char>(d.y + sourceSize.height/2, d.x + sourceSize.width/2);
        }
    }
}

void drawAffine(cv::Mat& image, cv::Mat& affine, const cv::Size kTemplateSize, const cv::Scalar& color=cv::Scalar(0, 255, 0), const int thickness=1) {
    std::vector<cv::Point> pointList;
    pointList.push_back( transform(affine, -kTemplateSize.width/2, -kTemplateSize.height/2) );
    pointList.push_back( transform(affine, -kTemplateSize.width/2, +kTemplateSize.height/2) );
    pointList.push_back( transform(affine, +kTemplateSize.width/2, +kTemplateSize.height/2) );
    pointList.push_back( transform(affine, +kTemplateSize.width/2, -kTemplateSize.height/2) );
    pointList.push_back( transform(affine, -kTemplateSize.width/2, -kTemplateSize.height/2) );

    cv::Point delta(image.size().width/2, image.size().height/2);
    for(int i=0; i<pointList.size()-1; i++) {
        cv::line( image,  pointList[i]+delta, pointList[i+1]+delta, color, thickness);
    }
}

int main(int argc, char* argv[]) {
    static struct option longOptions[] = {
        {"help", no_argument,       0, 'h'},
        {"path", required_argument, 0, 'p'},
    };

    std::string dataPath;

    int argopt, optionIndex=0;
    while( (argopt = getopt_long(argc, argv, "hp:", longOptions, &optionIndex)) != -1 ) {
        switch( argopt ) {
            case 'p':
                dataPath = std::string(optarg);
                break;
            case 'h':
            default:
                help(argv[0]);
                break;
        }
    }
    if( dataPath.size() == 0 ) {
        help(argv[0]);
    }

    const std::string imageFilenameTemplate = dataPath + "/im%03d.jpg";

    cv::Mat affine(3, 2, cv::DataType<float>::type);
    const cv::Size kTemplateImageSize(150, 150);
    cv::Mat templateImage(kTemplateImageSize, cv::DataType<unsigned char>::type);

    {
        std::string filename = instant::Utils::String::Format(imageFilenameTemplate.c_str(), 0);
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        copyTo(image, templateImage, affine);
        cv::imshow("template", templateImage);
    }
    std::vector<cv::Mat> gradient = calcGradient( templateImage);
    cv::imshow("gradient x", gradient[0] / (255.0) + 0.5);
    cv::imshow("gradient y", gradient[1] / (255.0) + 0.5);

    cv::Mat descent = calcSteepestDescent( gradient );
    cv::Mat descentImage = descent.reshape(0, kTemplateImageSize.height*6);
    cv::imshow("steeped descent", descentImage / (kTemplateImageSize.area()) + 0.5);

    cv::Mat hessianInv = (descent * descent.t()).inv();

    for(int i=0; i<200; i++){ 
        std::cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::" << std::endl;
        std::string filename = instant::Utils::String::Format(imageFilenameTemplate.c_str(), i);
        cv::Mat color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        drawAffine(color, affine, kTemplateImageSize, cv::Scalar(0, 255, 0), 3);

        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        for(int k=0; k<100; k++) {
            cv::Mat error = calcError(image, templateImage, affine);
            cv::imshow("error", error);

            cv::Mat errorVector = error.reshape(0, error.size().area());
            cv::Mat deltaAffine = hessianInv * (descent * errorVector);

            std::cout << "affine = " << affine << std::endl;
            affine = updateAffineCompositional(affine, inverseAffine(deltaAffine));
            std::cout << "affine = " << affine << std::endl;
            std::cout << "delta  = " << (deltaAffine) << std::endl;
            std::cout << "deltaI = " << inverseAffine(deltaAffine) << std::endl;

            drawAffine(color, affine, kTemplateImageSize, cv::Scalar(0, 0, 255));

            double normOfDelta = cv::norm( deltaAffine, cv::NORM_L2 );
            std::cout << normOfDelta << std::endl;
            if( normOfDelta < 0.01 )
                break;
        }
        
        cv::imshow("image", color);

        char ch = cv::waitKey(0);
        if( ch == 'q' || ch == 'Q' )
            break;
    }



    return 0;
}
