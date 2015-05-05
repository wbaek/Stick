#ifndef __EXCEPTIONS_INVALID_PARAMETERS_HPP__
#define __EXCEPTIONS_INVALID_PARAMETERS_HPP__

#include "exceptions/exception.hpp"

namespace Stick {
    class InvalidParameters : public Exception {
        public:
            InvalidParameters(
                    const char* filename, const int line, const char* functionName,
                    const std::string className="", const std::string& message="") throw() 
                : Exception(filename, line, functionName, className, message) { //noexcept
            }
    };
}

#endif //__EXCEPTIONS_INVALIDATE_PARAMETERS_HPP__
