#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

#include <utils/string.hpp>
#include <utils/filesystem.hpp>
#include <utils/others.hpp>
#include <opencv2/opencv.hpp>

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
        + affine.at<float>(0) * delta.at<float>(0) + affine.at<float>(2) * delta.at<float>(1);
    updated.at<float>(1) = affine.at<float>(1) + delta.at<float>(1)
        + affine.at<float>(1) * delta.at<float>(0) + affine.at<float>(3) * delta.at<float>(1);
    updated.at<float>(2) = affine.at<float>(2) + delta.at<float>(2)
        + affine.at<float>(0) * delta.at<float>(2) + affine.at<float>(2) * delta.at<float>(3);
    updated.at<float>(3) = affine.at<float>(3) + delta.at<float>(3)
        + affine.at<float>(1) * delta.at<float>(2) + affine.at<float>(3) * delta.at<float>(3);
    updated.at<float>(4) = affine.at<float>(4) + delta.at<float>(4)
        + affine.at<float>(0) * delta.at<float>(4) + affine.at<float>(2) * delta.at<float>(5);
    updated.at<float>(5) = affine.at<float>(5) + delta.at<float>(5)
        + affine.at<float>(1) * delta.at<float>(4) + affine.at<float>(3) * delta.at<float>(5);

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

    cv::Mat error = cv::Mat::zeros(templateImage.size(), cv::DataType<float>::type);
    for(int y=-templateImage.size().height/2; y<templateImage.size().height/2; y++) {
        for(int x=-templateImage.size().width/2; x<templateImage.size().width/2; x++) {
            cv::Point imagePoint = transform(affine, x, y) + delta;
            if( 0 <= imagePoint.x && imagePoint.x < image.size().width &&
                0 <= imagePoint.y && imagePoint.y < image.size().height ) {
            error.at<float>(cv::Point(x, y) + templateDelta)
                = ( (float)image.at<unsigned char>(imagePoint)
                        - (float)templateImage.at<unsigned char>(cv::Point(x, y) + templateDelta) );
            }
        }
    }
    return error;
}

template<typename DATA_TYPE>
void copyTo(const cv::Mat& image, cv::Mat& templateImage, const cv::Mat& affine) {
    cv::Point delta(image.size().width/2, image.size().height/2);
    cv::Point templateDelta(templateImage.size().width/2, templateImage.size().height/2);

    for(int y=-templateImage.size().height/2; y<templateImage.size().height/2; y++) {
        for(int x=-templateImage.size().width/2; x<templateImage.size().width/2; x++) {
            cv::Point imagePoint = transform(affine, x, y) + delta;
            imagePoint.x = std::min(std::max(0, imagePoint.x), image.size().width);
            imagePoint.y = std::min(std::max(0, imagePoint.y), image.size().height);
            templateImage.at<DATA_TYPE>(cv::Point(x, y) + templateDelta) = image.at<DATA_TYPE>(imagePoint);
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

void help(char* execute) {
    std::cerr << "usage: " << execute << " [-h] -p DATA_PATH [-t TEMPLATE_SIZE] [-e EPSILON_VALUE] [-k ITERATION] [-v]" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "\t-h, --help                     show this help message and exit" << std::endl;
    std::cerr << "\t-p, --path      DATA_PATH      set DATA_PATH" << std::endl;
    std::cerr << "\t-t, --template  SIZE           set TEMPLATE_SIZE (default:150)" << std::endl;
    std::cerr << "\t-g, --gaussian  KERNAL_SIZE    set GAUSSIAN_KERNAL_SIZE (default:15)" << std::endl;
    std::cerr << "\t-e, --epsilon   EPSILON_VALUE  set EPSILON_VALUE (default:0.05)" << std::endl;
    std::cerr << "\t-k, --iteration ITERATION      set max ITERATION per update (default:100)" << std::endl;
    std::cerr << "\t-v, --verbose                  verbose" << std::endl;
    exit(-1);
}

int main(int argc, char* argv[]) {
    static struct option longOptions[] = {
        {"help",      no_argument,       0, 'h'},
        {"path",      required_argument, 0, 'p'},
        {"template",  required_argument, 0, 't'},
        {"gaussian",  required_argument, 0, 'g'},
        {"epsilon",   required_argument, 0, 'e'},
        {"iteration", required_argument, 0, 'k'},
        {"verboase",  no_argument,       0, 'v'},
    };

    std::string dataPath;
    int templateSize = 150;
    float epsilon = 0.05;
    int iteration = 100;
    int gaussianBlurSize = 15;
    bool verbose = 0;

    int argopt, optionIndex=0;
    while( (argopt = getopt_long(argc, argv, "hp:t:g:e:k:v", longOptions, &optionIndex)) != -1 ) {
        switch( argopt ) {
            case 'p':
                dataPath = std::string(optarg);
                break;
            case 't':
                instant::Utils::String::ToPrimitive<int>(optarg, templateSize);
                break;
            case 'e':
                instant::Utils::String::ToPrimitive<float>(optarg, epsilon);
                break;
            case 'g':
                instant::Utils::String::ToPrimitive<int>(optarg, gaussianBlurSize);
                break;
            case 'k':
                instant::Utils::String::ToPrimitive<int>(optarg, iteration);
                break;
            case 'v':
                verbose = true;
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

    double current = instant::Utils::Others::GetMilliSeconds();
    std::vector<std::string> filelist;
    instant::Utils::Filesystem::GetFileNames(dataPath, filelist);

    // pre-computing
    cv::Mat affine = cv::Mat::zeros(3, 2, cv::DataType<float>::type);
    const cv::Size kTemplateImageSize(templateSize, templateSize);
    cv::Mat templateImage(kTemplateImageSize, cv::DataType<unsigned char>::type);
    cv::Mat generatedImage(kTemplateImageSize, cv::DataType<unsigned char>::type);
    std::vector<cv::Mat> gradient;
    {
        std::string filename = filelist[0];
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
        cv::GaussianBlur(image, image, cv::Size(gaussianBlurSize, gaussianBlurSize), 0);
        copyTo<unsigned char>(image, templateImage, affine);
        //cv::imshow("template", templateImage);

        std::vector<cv::Mat> imageGradient = calcGradient(image);
        gradient.resize( imageGradient.size() );
        for(unsigned int i=0; i<imageGradient.size(); i++) {
            gradient[i] = cv::Mat::zeros(templateImage.size(), cv::DataType<float>::type);
            copyTo<float>(imageGradient[i], gradient[i], affine);
        }
    }

    cv::Mat descent = calcSteepestDescent( gradient );
    cv::Mat descentImage = descent.reshape(0, kTemplateImageSize.height*6);
    cv::Mat hessianInv = (descent * descent.t()).inv();

    // active computing
    for(std::string& filename : filelist){
        std::vector<cv::Mat> affineList;        
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        double startTime = instant::Utils::Others::GetMilliSeconds();
        cv::GaussianBlur(image, image, cv::Size(gaussianBlurSize, gaussianBlurSize), 0);

        double normOfDelta, normOfError;
        int k=0;
        for(k=0; k<iteration; k++) {
            cv::Mat error = calcError(image, templateImage, affine);

            cv::Mat errorVector = error.reshape(0, error.size().area());
            cv::Mat deltaAffine = hessianInv * (descent * errorVector);
            affine = updateAffineCompositional(affine, inverseAffine(deltaAffine));
            affineList.push_back( affine );

            normOfDelta = cv::norm( deltaAffine, cv::NORM_L2 );
            normOfError = cv::norm( errorVector, cv::NORM_L2 ) / (double)errorVector.size().area();
            if( normOfDelta < epsilon )
                break;
        }
        double endTime = instant::Utils::Others::GetMilliSeconds();

        //copyTo<unsigned char>(image, generatedImage, affine);
        //cv::imshow("generated", generatedImage);

        // draw tracking result
        cv::Mat color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        for(auto affineIter : affineList) {
            drawAffine(color, affineIter, kTemplateImageSize, cv::Scalar(0, 0, 255), 1);
        }
        drawAffine(color, affine, kTemplateImageSize, cv::Scalar(0, 255, 0), 2);
        cv::imshow("image", color);
        char ch = cv::waitKey(1);
        if( ch == 'q' || ch == 'Q' )
            break;

        if(verbose) {
            std::string message =
                instant::Utils::String::Format("%s: iter=%03d, delta=%.3f, error=%.3f, time=%.3fsec",
                        filename.c_str(), k, normOfDelta, normOfError, (endTime-startTime)/1000.0);
            std::cout << message << std::endl;
        }
    }

    return 0;
}
