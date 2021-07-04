#include "emptyNN/Types.hpp"
namespace emptyNN {

    std::string DeviceToStr_(Device d) {
        switch (d)
            {
                case CPU: return "CPU";
                case CPU_RVV: return "CPU_RVV";
                case CPU_SVE: return "CPU_SVE";
                case GPU: return "GPU";
                default: return "INVALID";
            }
    }


    DeviceNotAllowed::DeviceNotAllowed(Device d): d(d){};
    const char* DeviceNotAllowed::what() const throw() {
        std::string msg("Device: ");
        msg += DeviceToStr_(d) + " not allowed";
        return msg.c_str();
    }

}    



