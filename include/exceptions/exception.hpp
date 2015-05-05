#ifndef __EXCEPTIONS_EXCEPTION_HPP__
#define __EXCEPTIONS_EXCEPTION_HPP__

#include <cxxabi.h>

#include "utils/string.hpp"
#include "utils/type.hpp"

#define MakeException(EXCEPTION_NAME, CLASS_NAME, MESSAGE) EXCEPTION_NAME(__FILE__, __LINE__, __FUNCTION__, CLASS_NAME, MESSAGE)
#define MakeClassException(EXCEPTION_NAME, MESSAGE) MakeException(EXCEPTION_NAME, this->getName(), MESSAGE)
namespace Stick {
    class Exception {
        protected:
            std::string filename;
            int line;
            std::string className;
            std::string functionName;
            std::string message;

        public:
            Exception(
                    const char* filename, const int line, const char* functionName,
                    const std::string className="", const std::string& message="") throw() { //noexcept
                this->filename = std::string(filename);
                this->line = line;
                this->className = std::string(className);
                this->functionName = std::string(functionName);
                this->message = message;
            }
            virtual ~Exception() throw() { //noexcept
            }
            
            virtual const std::string GetName() const {
                std::string className = instant::Utils::Type::GetTypeName(this);
                return instant::Utils::String::Replace(className, "Stick::", "");
            }

            virtual const std::string what() const throw() { //noexcept
                const std::string exceptionMessageTemplate = std::string() +
                    "[${exception_name}] " +
                    "[${filename}(${line})] " +
                    "[${class_name}::${function_name}] " +
                    "${message}";

                std::map<std::string, std::string> bindStrings;
                bindStrings["exception_name"] = this->GetName();
                bindStrings["filename"]       = this->filename;
                bindStrings["line"]           = instant::Utils::String::ToString( this->line );
                bindStrings["class_name"]     = this->className;
                bindStrings["function_name"]  = this->functionName;
                bindStrings["message"]        = this->message;

                return instant::Utils::String::Substitute(exceptionMessageTemplate, bindStrings);
            }
    };
}

#endif //__EXCEPTIONS_EXCEPTION_HPP__
