#include <cmath>
#include <algorithm>

extern "C" {

#ifdef _WIN32
    __declspec(dllexport) void calculate_portfolio_step(
#else
    void calculate_portfolio_step(
#endif
        double target_pos,
        double mid_price,
        double prev_price,
        double spread,
        double balance,
        double entry_price,
        double current_pos,
        double leverage,
        double margin_pct,
        double multiplier,
        double kyle_lambda,
        double slippage,
        double* out_realized_pnl,
        double* out_unrealized_pnl,
        double* out_new_position,
        double* out_new_balance,
        double* out_new_entry_price,
        int* out_margin_called
    ) {
        // Initialize outputs
        *out_realized_pnl = 0.0;
        *out_unrealized_pnl = 0.0;
        *out_new_position = current_pos;
        *out_new_balance = balance;
        *out_new_entry_price = entry_price;
        *out_margin_called = 0;

        double active_pos = current_pos;
        double active_balance = balance;
        double active_entry = entry_price;

        // 1. Calculate price transition delta for current position
        double price_delta = mid_price - prev_price;
        double delta_unrealized = 0.0;
        if (std::abs(active_pos) > 1e-8) {
            delta_unrealized = active_pos * price_delta * multiplier;
        }

        // 2. Process transaction and state changes if position weight is adjusted
        double trade_size = target_pos - active_pos;
        double tx_cost = 0.0;

        if (std::abs(trade_size) > 1e-8) {
            // Apply Kyle's Lambda market impact cost model
            tx_cost = std::abs(trade_size) * multiplier * ((spread / 2.0) + (kyle_lambda * slippage));
            active_balance -= tx_cost;

            if (std::abs(active_pos) < 1e-8) {
                // Open new position
                active_pos = target_pos;
                active_entry = mid_price;
            }
            else if ((active_pos > 0.0 && target_pos > 0.0) || (active_pos < 0.0 && target_pos < 0.0)) {
                // Same direction trade
                if (std::abs(target_pos) > std::abs(active_pos)) {
                    // Increase position size: volume-weighted entry price
                    active_entry = ((active_pos * active_entry) + (trade_size * mid_price)) / target_pos;
                    active_pos = target_pos;
                }
                else {
                    // Reduce position size (partial exit): realize partial PnL
                    double closed_size = std::abs(trade_size);
                    double realized_trade_pnl = 0.0;
                    if (active_pos > 0.0) {
                        realized_trade_pnl = closed_size * (mid_price - active_entry) * multiplier;
                    } else {
                        realized_trade_pnl = closed_size * (active_entry - mid_price) * multiplier;
                    }
                    *out_realized_pnl = realized_trade_pnl;
                    active_balance += realized_trade_pnl;
                    active_pos = target_pos;
                }
            }
            else {
                // Reverse or completely close position
                // Realize full PnL on existing position
                double closed_size = std::abs(active_pos);
                double realized_trade_pnl = 0.0;
                if (active_pos > 0.0) {
                    realized_trade_pnl = closed_size * (mid_price - active_entry) * multiplier;
                } else {
                    realized_trade_pnl = closed_size * (active_entry - mid_price) * multiplier;
                }
                *out_realized_pnl = realized_trade_pnl;
                active_balance += realized_trade_pnl;

                // Open reversed new position
                active_pos = target_pos;
                if (std::abs(target_pos) > 1e-8) {
                    active_entry = mid_price;
                } else {
                    active_entry = 0.0;
                }
            }
        }

        // 3. Compute final unrealized PnL of new active position
        double active_unrealized = 0.0;
        if (std::abs(active_pos) > 1e-8) {
            active_unrealized = active_pos * (mid_price - active_entry) * multiplier;
        }

        // 4. Perform margin requirement gating checks
        double equity = active_balance + active_unrealized;
        double margin_required = 0.0;

        if (std::abs(active_pos) > 1e-8) {
            margin_required = std::abs(active_pos) * mid_price * multiplier * margin_pct;
        }

        // 50% margin call closeout rule
        if (margin_required > 0.0 && equity <= (margin_required * 0.5)) {
            *out_margin_called = 1;
            *out_new_position = 0.0;
            *out_new_balance = std::max(0.0, equity);
            *out_new_entry_price = 0.0;
            *out_unrealized_pnl = 0.0;
            // Realized loss is the absolute drop in balance
            *out_realized_pnl += (std::max(0.0, equity) - balance);
        } else {
            *out_new_position = active_pos;
            *out_new_balance = active_balance;
            *out_new_entry_price = active_entry;
            *out_unrealized_pnl = active_unrealized;
        }
    }


#ifdef _WIN32
    __declspec(dllexport) void calculate_portfolio_loop(
#else
    void calculate_portfolio_loop(
#endif
        const double* target_positions,
        const double* mid_prices,
        const double* spreads,
        const double* kyle_lambdas,
        int n,
        double initial_balance,
        double leverage,
        double margin_pct,
        double multiplier,
        double slippage,
        double* out_balances,
        double* out_positions,
        double* out_entry_prices,
        double* out_realized_pnls,
        double* out_unrealized_pnls,
        int* out_margin_called_steps
    ) {
        double current_balance = initial_balance;
        double current_position = 0.0;
        double current_entry_price = 0.0;

        out_balances[0] = initial_balance;
        out_positions[0] = 0.0;
        out_entry_prices[0] = 0.0;
        out_realized_pnls[0] = 0.0;
        out_unrealized_pnls[0] = 0.0;
        out_margin_called_steps[0] = 0;

        bool margin_called = false;

        for (int i = 0; i < n - 1; ++i) {
            if (margin_called) {
                out_balances[i + 1] = current_balance;
                out_positions[i + 1] = 0.0;
                out_entry_prices[i + 1] = 0.0;
                out_realized_pnls[i + 1] = 0.0;
                out_unrealized_pnls[i + 1] = 0.0;
                out_margin_called_steps[i + 1] = 1;
                continue;
            }

            double target_pos = target_positions[i];
            double mid_price = mid_prices[i + 1];
            double prev_price = mid_prices[i];
            double spread = spreads[i + 1];
            double kyle_lambda = kyle_lambdas[i + 1];

            double out_realized = 0.0;
            double out_unrealized = 0.0;
            double out_pos = 0.0;
            double out_bal = 0.0;
            double out_entry = 0.0;
            int out_mcall = 0;

            calculate_portfolio_step(
                target_pos,
                mid_price,
                prev_price,
                spread,
                current_balance,
                current_entry_price,
                current_position,
                leverage,
                margin_pct,
                multiplier,
                kyle_lambda,
                slippage,
                &out_realized,
                &out_unrealized,
                &out_pos,
                &out_bal,
                &out_entry,
                &out_mcall
            );

            out_balances[i + 1] = out_bal;
            out_positions[i + 1] = out_pos;
            out_entry_prices[i + 1] = out_entry;
            out_realized_pnls[i + 1] = out_realized;
            out_unrealized_pnls[i + 1] = out_unrealized;
            out_margin_called_steps[i + 1] = out_mcall;

            current_balance = out_bal;
            current_position = out_pos;
            current_entry_price = out_entry;

            if (out_mcall == 1) {
                margin_called = true;
            }
        }
    }
}
