// Rcpp
#include <Rcpp.h>
using namespace Rcpp;

// other
#include <limits>
#include <string>
#include <stdlib.h>

// deepbayes
#include "cv_error_function.hpp"

// [[Rcpp::export]]
List brute_force(NumericMatrix xPoints_mat,
                 int k,
                 std::vector<std::string> param_prefixes,
                 std::string solver_dir,
                 std::string log_dir) {

    const int nPoints = xPoints_mat.ncol();
    const int n = xPoints_mat.nrow();
    user_function_data ui_data;
    ui_data.k = k;
    ui_data.param_prefixes = param_prefixes;
    ui_data.solver_dir = solver_dir;
    ui_data.log_dir = log_dir;

    double min_f = std::numeric_limits<double>::infinity();
    NumericVector min_x;
    for (int i = 0; i < nPoints; ++i) {
        NumericVector col = xPoints_mat(_, i);
        double this_f;
        this_f = cv_error_function(n, col.begin(), NULL, &ui_data);
        std::cout << "x = " << col << ": f = " << this_f << std::endl;
        if (this_f < min_f) {
            min_x = col;
            min_f = this_f;
        }
    }

    std::cout << "min x = " << min_x <<": achieves f = " << min_f << std::endl;

    NumericVector f = NumericVector::create(min_f);
    List z = List::create(min_x, f);
    return(z);
}