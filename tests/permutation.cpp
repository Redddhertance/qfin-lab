#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <numeric>

namespace py = pybind11;

py::array_t<double> run_permutations_fast(py::array_t<double> cross_pnl, int n_trials) {
    py::buffer_info pnl_buf = cross_pnl.request();

    int n_days = pnl_buf.shape[0]; // matrix is (days*days)
    double* pnl_ptr = static_cast<double*>(pnl_buf.ptr);

    // output array: (n_trials x n_days)
    auto result = py::array_t<double>({n_trials, n_days});
    py::buffer_info res_buf = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    std::vector<int> days(n_days);
    std::iota(days.begin(), days.end(), 0);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int trial = 0; trial < n_trials; ++trial) {
        std::shuffle(days.begin(), days.end(), rng);

        double cum_equity = 1.0;
        for (int t = 0; t < n_days; ++t) {
            int random_t = days[t]; 
            
            // O(1) get pnl from using weights t on days t
            double daily_pnl = pnl_ptr[random_t * n_days + t];
            
            cum_equity *= (1.0 + daily_pnl);
            res_ptr[trial * n_days + t] = cum_equity;
        }
    }

    return result;
}

PYBIND11_MODULE(gaka_core, m) {
    m.def("run_permutations_fast", &run_permutations_fast, "Lightning fast cross-matrix permutation test");
}