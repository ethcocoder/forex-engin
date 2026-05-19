#include <cmath>

extern "C" {
    /**
     * Highly optimized 2D Kalman Filter routine.
     * Computes recursive state transition and observation updates directly in C++.
     * Prevents python loop interpreter overhead.
     *
     * State:     x = [price, velocity]^T
     * Transition: A = [[1.0, 1.0], [0.0, 1.0]]
     * Observation: H = [[1.0, 0.0]]
     * Process Noise Q: [[qp, 0.0], [0.0, qv]]
     * Observation Noise R: [[r]]
     */
#ifdef _WIN32
    __declspec(dllexport) void kalman_filter_2d(
#else
    void kalman_filter_2d(
#endif
        const double* close,
        int n,
        double qp,
        double qv,
        double r,
        double* filtered_price,
        double* velocity_estimate
    ) {
        if (n <= 0) return;

        // Initialize state vector: x = [initial_close, 0.0]^T
        double x0 = close[0];
        double x1 = 0.0;

        // Initialize covariance matrix: P = [[1.0, 0.0], [0.0, 1.0]]
        double P00 = 1.0;
        double P01 = 0.0;
        double P10 = 0.0;
        double P11 = 1.0;

        filtered_price[0] = x0;
        velocity_estimate[0] = x1;

        for (int k = 1; k < n; ++k) {
            double z = close[k];

            // 1. Predict Step
            // xp = A * x
            double xp0 = x0 + x1;
            double xp1 = x1;

            // P_pred = A * P * A.T + Q
            double Pp00 = P00 + P01 + P10 + P11 + qp;
            double Pp01 = P01 + P11;
            double Pp10 = P10 + P11;
            double Pp11 = P11 + qv;

            // 2. Update Step
            // Innovation: y = z - H * xp
            double y = z - xp0;

            // Innovation covariance: S = H * P_pred * H.T + R
            double S = Pp00 + r;

            // Avoid division by zero
            if (std::abs(S) < 1e-15) {
                S = (S >= 0) ? 1e-15 : -1e-15;
            }

            // Kalman Gain: K = P_pred * H.T * S^-1
            double K0 = Pp00 / S;
            double K1 = Pp10 / S;

            // Update State: x = xp + K * y
            x0 = xp0 + K0 * y;
            x1 = xp1 + K1 * y;

            // Update Covariance: P = (I - K * H) * P_pred
            P00 = (1.0 - K0) * Pp00;
            P01 = (1.0 - K0) * Pp01;
            P10 = Pp10 - K1 * Pp00;
            P11 = Pp11 - K1 * Pp01;

            // Store output
            filtered_price[k] = x0;
            velocity_estimate[k] = x1;
        }
    }
}
