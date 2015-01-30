#pragma once

namespace Minimizer {
    class Minimizer {
        private:
            Minimizer() {
            };
            virtual ~Minimizer() {
            }

        public:
            virtual double minimize() = 0;
    };
}
