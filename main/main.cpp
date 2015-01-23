#include <iostream>
#include <cstdlib>
#include <getopt.h>

void help(char* execute) {
    std::cerr << "usage: " << execute << " [-h]" << std::endl;
    std::cerr << "" << std::endl;
    std::cerr << "\t-h, --help              show this help message and exit" << std::endl;
    exit(-1);
}

int main(int argc, char* argv[]) {
    static struct option longOptions[] = {
        {"help", no_argument, 0, 'h'},
    };

    int argopt, optionIndex=0;
    while( (argopt = getopt_long(argc, argv, "h", longOptions, &optionIndex)) != -1 ) {
        switch( argopt ) {
            case 'h':
                help(argv[0]);
                break;
        }
    }

    return 0;
}
