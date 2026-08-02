#include <cmath>
#include <numeric>
#include <vector>

extern "C" {

/**
 * Calculates the z-score of the recent rolling volatility against the historical volatility.
 * Used by the Anti-Fragile Risk Engine to detect 'Black Swan' regimes in nanoseconds.
 */
double calculate_vol_z_score(const double* historical_returns, int n) {
    if (n < 50) {
        return 0.0;
    }
    
    // 1. Compute historical mean and variance
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += historical_returns[i];
    }
    double hist_mean = sum / n;
    
    double sq_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = historical_returns[i] - hist_mean;
        sq_sum += diff * diff;
    }
    double hist_vol = std::sqrt(sq_sum / n);
    
    // 2. Compute recent mean and variance (last 20 items)
    double sum_20 = 0.0;
    for (int i = n - 20; i < n; ++i) {
        sum_20 += historical_returns[i];
    }
    double recent_mean = sum_20 / 20.0;
    
    double sq_sum_20 = 0.0;
    for (int i = n - 20; i < n; ++i) {
        double diff = historical_returns[i] - recent_mean;
        sq_sum_20 += diff * diff;
    }
    double recent_vol = std::sqrt(sq_sum_20 / 20.0);
    
    // 3. Compute std of diff of historical returns (returns_t - returns_{t-1})
    double sum_diff = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        sum_diff += (historical_returns[i+1] - historical_returns[i]);
    }
    double diff_mean = sum_diff / (n - 1);
    
    double sq_sum_diff = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double diff = (historical_returns[i+1] - historical_returns[i]) - diff_mean;
        sq_sum_diff += diff * diff;
    }
    double diff_std = std::sqrt(sq_sum_diff / (n - 1));
    
    // 4. Return Z-score
    return (recent_vol - hist_vol) / (diff_std + 1e-6);
}

}
