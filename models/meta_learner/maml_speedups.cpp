#include <cmath>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

EXPORT void maml_inner_loop_update(double* params, const double* gradients, double learning_rate, int param_count, int num_steps) {
    for (int step = 0; step < num_steps; ++step) {
        int offset = step * param_count;
        for (int i = 0; i < param_count; ++i) {
            params[i] -= learning_rate * gradients[offset + i];
        }
    }
}

EXPORT void compute_linear_forward_and_gradient(const double* X, const double* y, const double* weights, const double* bias, int n_samples, int d_feat, double* out_loss, double* out_grad_w, double* out_grad_b) {
    double total_loss = 0.0;
    for (int j = 0; j < d_feat; ++j) {
        out_grad_w[j] = 0.0;
    }
    out_grad_b[0] = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        double y_hat = bias[0];
        for (int j = 0; j < d_feat; ++j) {
            y_hat += X[i * d_feat + j] * weights[j];
        }
        
        double error = y_hat - y[i];
        total_loss += error * error;
        
        double grad_base = 2.0 * error / n_samples;
        
        for (int j = 0; j < d_feat; ++j) {
            out_grad_w[j] += grad_base * X[i * d_feat + j];
        }
        out_grad_b[0] += grad_base;
    }
    out_loss[0] = total_loss / n_samples;
}

}
