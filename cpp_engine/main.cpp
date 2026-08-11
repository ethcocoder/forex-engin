#include "Elite10xTradingEngine.hpp"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  INITIALIZING ELITE10X-PR HARDENED C++ CORE      " << std::endl;
    std::cout << "==================================================" << std::endl;

    Elite10x::ChaosTestRunner chaos_runner(100000.0);
    chaos_runner.run_comprehensive_chaos_suite();

    return 0;
}
