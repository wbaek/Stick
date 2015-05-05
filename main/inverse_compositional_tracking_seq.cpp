#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

#include <utils/string.hpp>
#include <utils/filesystem.hpp>
#include <utils/others.hpp>
#include <opencv2/opencv.hpp>

#include <tracker/inverse_compositional.hpp>
#include <model/homography.hpp>

void help(char* execute) {
    std::cerr << "usage: " << execute << " [-h] -p DATA_PATH [-t TEMPLATE_SIZE] [-g GAUSSIAN_KERNAL_SIZE] [-e EPSILON_VALUE] [-k ITERATION] [-b] [-v]" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "\t-h, --help                           show this help message and exit" << std::endl;
    std::cerr << "\t-p, --path      DATA_PATH            set DATA_PATH" << std::endl;
    std::cerr << "\t-t, --template  SIZE                 set TEMPLATE_SIZE (default:200)" << std::endl;
    std::cerr << "\t-g, --gaussian  GAUSSIAN_KERNAL_SIZE set GAUSSIAN_KERNAL_SIZE (default:21)" << std::endl;
    std::cerr << "\t-e, --epsilon   EPSILON_VALUE        set EPSILON_VALUE (default:0.05)" << std::endl;
    std::cerr << "\t-k, --iteration ITERATION            set max ITERATION per update (default:100)" << std::endl;
    std::cerr << "\t-b, --break;                         break wait iter" << std::endl;
    std::cerr << "\t-v, --verbose                        verbose" << std::endl;
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
        {"break;",    no_argument,       0, 'b'},
        {"verboase",  no_argument,       0, 'v'},
    };

    std::string dataPath;
    int templateSize = 200;
    float epsilon = 0.05;
    int iteration = 100;
    int gaussianBlurSize = 21;
    bool breakIter = false;
    bool verbose = false;

    int argopt, optionIndex=0;
    while( (argopt = getopt_long(argc, argv, "hp:t:g:e:k:bv", longOptions, &optionIndex)) != -1 ) {
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
            case 'b':
                breakIter = true;
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

    // initialze
    Stick::InverseCompositional* tracker = new Stick::InverseCompositional(new Stick::Homography(), epsilon, iteration);
    cv::Mat image = cv::imread(filelist[0], CV_LOAD_IMAGE_GRAYSCALE);
    cv::GaussianBlur(image, image, cv::Size(gaussianBlurSize, gaussianBlurSize), gaussianBlurSize/2.0, gaussianBlurSize/2.0);
    tracker->calculateTransformedImage(image, cv::Size(templateSize, templateSize));
    tracker->setTemplateImage( tracker->getTransformedImage() );
    tracker->initialize();

    // active computing
    for(std::string& filename : filelist){
        cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

        double startTime = instant::Utils::Others::GetMilliSeconds();
        cv::GaussianBlur(image, image, cv::Size(gaussianBlurSize, gaussianBlurSize), gaussianBlurSize/2.0, gaussianBlurSize/2.0);
        tracker->track(image);
        double endTime = instant::Utils::Others::GetMilliSeconds();

        // draw result
        cv::Mat color = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        if( verbose ) {
            std::vector<cv::Mat> pose = tracker->getPoseTrace();
            Stick::Homography h;
            for(cv::Mat& p : pose) {
                h.set(p);
                h.draw(color, cv::Size(templateSize, templateSize), CV_RGB(255, 0, 0), 1);
            }
            
            cv::imshow("template", tracker->getTemplateImage());
            cv::imshow("transformed", tracker->getTransformedImage());
            cv::imshow("error", tracker->getErrorImage()/(255.0/2.0) + 0.5);
        }
        tracker->getModel()->draw(color, cv::Size(templateSize, templateSize), CV_RGB(0, 255, 0), 3); 
        cv::imshow("image", color);

        int waitTime = 30 - (int)(startTime - endTime);
        waitTime = waitTime <= 0 ? 1 : waitTime;
        waitTime = breakIter ? 0 : waitTime;
        char ch = cv::waitKey(waitTime);
        if( ch == 'q' || ch == 'Q' )
            break;

        if(verbose) {
            std::string message =
                instant::Utils::String::Format("%s: time=%.3fsec",
                        filename.c_str(),
                        (endTime-startTime)/1000.0);
            std::cout << message << std::endl;
            std::cout << tracker->getLogString() << std::endl;
        }
    }

    return 0;
}
