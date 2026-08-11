#include "Elite10xTradingEngine.hpp"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  INITIALIZING ELITE10X-PR ULTRA C++ TRADING CORE " << std::endl;
    std::cout << "==================================================" << std::endl;

    Elite10x::Elite10xSystem system(100000.0);

    // Generate synthetic tick stream across major currency pairs (EURUSD, GBPUSD, USDJPY, AUDUSD)
    std::vector<std::string> pairs = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD"};
    std::vector<Elite10x::MarketTick> ticks;
    
    std::mt19937 gen(1337);
    std::uniform_real_distribution<double> price_dist(1.0500, 1.3500);
    std::uniform_real_distribution<double> vol_dist(0.0005, 0.0025);

    double base_time = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < 5000; ++i) {
        std::string pair = pairs[i % pairs.size()];
        double mid = price_dist(gen);
        double spread = 1.2;
        double vol = vol_dist(gen);
        ticks.push_back({
            pair,
            base_time + i * 1.0,
            mid - (spread * 0.00005),
            mid + (spread * 0.00005),
            mid,
            150000.0,
            spread,
            vol
        });
    }

    system.run_simulation_cycle(ticks);

    return 0;
}
