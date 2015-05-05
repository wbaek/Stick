#ifndef __EXCEPTIONS_NOT_INITIALIZED_HPP__
#define __EXCEPTIONS_NOT_INITIALIZED_HPP__

#include "exceptions/exception.hpp"

namespace Stick {
    class NotInitialized : public Exception {
        public:
			//using Exception::Exception; // available in GCC 4.8+, -std=c++11
			NotInitialized(
                    const char* filename, const int line, const char* functionName,
                    const std::string className="", const std::string& message="") throw() 
                : Exception(filename, line, functionName, className, message) { //noexcept
            }
	};
}

#endif //__EXCEPTIONS_NOT_INITIALIZED_HPP__
