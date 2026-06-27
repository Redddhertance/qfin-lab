#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <numeric>

namespace py = pybind11;

// Takes numpy arrays of weights and returns, outputs a 2D numpy array of equity curves
py::array_t<double> run_permutations_fast(py::array_t<double> weights, py::array_t<double> returns, int n_trials) {
    py::buffer_info w_buf = weights.request();
    py::buffer_info r_buf = returns.request();

    int n_days = w_buf.shape[0];
    int n_assets = w_buf.shape[1];

    // python pointers
    double* w_ptr = static_cast<double*>(w_buf.ptr);
    double* r_ptr = static_cast<double*>(r_buf.ptr);

    // memory for output array (n_trials x n_days)
    auto result = py::array_t<double>({n_trials, n_days});
    py::buffer_info res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // [0,1,2...n-1]
    std::vector<int> days(n_days);
    std::iota(days.begin(), days.end(), 0);

    //rand num gen
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int trial = 0; trial < n_trials; ++trial) {
        //shuffles  indices instead of matrix
        std::shuffle(days.begin(), days.end(), rng);

        double cum_equity = 1.0;
        for (int t = 0; t < n_days; ++t) {
            int random_t = days[t]; 
            double daily_pnl = 0.0;
            
            //portfolio return for the day using the randomized weight row
            for (int a = 0; a < n_assets; ++a) {
                daily_pnl += w_ptr[random_t * n_assets + a] * r_ptr[t * n_assets + a];
            }
            
            cum_equity *= (1.0 + daily_pnl);
            res_ptr[trial * n_days + t] = cum_equity; // output array (moved from vector)
        }
    }

    return result;
}

// bind to python
PYBIND11_MODULE(gaka_core, m) {
    m.def("run_permutations_fast", &run_permutations_fast, "Fast C++ permutation test");
}