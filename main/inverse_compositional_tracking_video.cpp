#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>

#include <utils/string.hpp>
#include <utils/filesystem.hpp>
#include <utils/others.hpp>
#include <opencv2/opencv.hpp>

void help(char* execute) {
    std::cerr << "usage: " << execute << " [-h] -p DATA_PATH [-t TEMPLATE_SIZE] [-g GAUSSIAN_KERNAL_SIZE] [-e EPSILON_VALUE] [-k ITERATION] [-v]" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "\t-h, --help                           show this help message and exit" << std::endl;
    std::cerr << "\t-t, --template  SIZE                 set TEMPLATE_SIZE (default:300)" << std::endl;
    std::cerr << "\t-g, --gaussian  GAUSSIAN_KERNAL_SIZE set GAUSSIAN_KERNAL_SIZE (default:49)" << std::endl;
    std::cerr << "\t-e, --epsilon   EPSILON_VALUE        set EPSILON_VALUE (default:0.05)" << std::endl;
    std::cerr << "\t-k, --iteration ITERATION            set max ITERATION per update (default:100)" << std::endl;
    std::cerr << "\t-v, --verbose                        verbose" << std::endl;
    exit(-1);
}

int main(int argc, char* argv[]) {
    static struct option longOptions[] = {
        {"help",      no_argument,       0, 'h'},
        {"template",  required_argument, 0, 't'},
        {"gaussian",  required_argument, 0, 'g'},
        {"epsilon",   required_argument, 0, 'e'},
        {"iteration", required_argument, 0, 'k'},
        {"verboase",  no_argument,       0, 'v'},
    };

    int templateSize = 300;
    float epsilon = 0.05;
    int iteration = 100;
    int gaussianBlurSize = 49;
    bool verbose = 0;

    int argopt, optionIndex=0;
    while( (argopt = getopt_long(argc, argv, "ht:g:e:k:v", longOptions, &optionIndex)) != -1 ) {
        switch( argopt ) {
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

    cv::VideoCapture capture(0);

    bool processing = true, tracking = false;
    while(processing) {
        cv::Mat image;
        capture >> image;

        double startTime = instant::Utils::Others::GetMilliSeconds();
        if(tracking) {
            cv::Mat gray;
            cv::cvtColor(image, gray, CV_BGR2GRAY);
        }
        double endTime = instant::Utils::Others::GetMilliSeconds();

        // draw result
        if( verbose ) {
        }
        
        cv::imshow("image", image);
        char ch = cv::waitKey(1);
        switch( ch ) {
            case 'q':
            case 'Q':
                processing = false;
                break;
            case 'r':
            case 'R':
                {
                }
                break;
            case ' ':
                {
                }
                break;
        }
        
        if(verbose) {
            std::string message =
                instant::Utils::String::Format("time=%.3fsec",
                        (endTime-startTime)/1000.0);
            std::cout << message << std::endl;
        }
    }

    return 0;
}
