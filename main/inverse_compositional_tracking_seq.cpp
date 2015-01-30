#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

#include <utils/string.hpp>
#include <utils/filesystem.hpp>
#include <utils/others.hpp>
#include <opencv2/opencv.hpp>

#include <tracking/affine.hpp>
#include <tracking/inverse_compositional.hpp>

cv::Point transform(const cv::Mat& pose, int x, int y) {
    Tracking::Affine model;
    model.setModel(pose);
    return model.transform(cv::Point(x, y));
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

    std::vector<std::string> filelist;
    instant::Utils::Filesystem::GetFileNames(dataPath, filelist);

    Tracking::InverseCompositional<Tracking::Affine> tracker;
    tracker.setMaxIteration(iteration);
    tracker.setGaussianKernalSize(gaussianBlurSize);
    tracker.setEpsilon(epsilon);

    // pre-computing
    const cv::Size kTemplateImageSize(templateSize, templateSize);
    {
        std::string filename = filelist[0];
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        cv::Mat affine = cv::Mat::zeros(3, 2, cv::DataType<float>::type);
        tracker.setPose(affine);
        cv::Mat templateImage = tracker.getTrackedImage(image, kTemplateImageSize);
        tracker.setTemplate(templateImage);
    }

    // active computing
    for(std::string& filename : filelist){
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        double startTime = instant::Utils::Others::GetMilliSeconds();
        tracker.track(image);
        double endTime = instant::Utils::Others::GetMilliSeconds();

        // draw tracking result
        cv::Mat color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        if( verbose ) {
            for(auto affineIter : tracker.getUpdatePoseList()) {
                drawAffine(color, affineIter, kTemplateImageSize, cv::Scalar(0, 0, 255), 1);
            }
        }
        cv::Mat affine = tracker.getPose();
        drawAffine(color, affine, kTemplateImageSize, cv::Scalar(0, 255, 0), 2);
        cv::imshow("image", color);
        char ch = cv::waitKey(1);
        if( ch == 'q' || ch == 'Q' )
            break;

        if(verbose) {
            std::string message =
                instant::Utils::String::Format("%s: iter=%03d, delta=%.3f, error=%02.3f, time=%.3fsec",
                        filename.c_str(),
                        tracker.getUpdateIteration(),
                        tracker.getDeltaPose(),
                        tracker.getError(),
                        (endTime-startTime)/1000.0);
            std::cout << message << std::endl;
        }
    }

    return 0;
}