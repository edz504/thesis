// BayesOpt
#include <bayesopt/bayesopt.h>
#include <bayesopt/bayesopt.hpp>

// Rcpp
#include <Rcpp.h>
using namespace Rcpp;

// other
#include <string>
#include <stdlib.h>

// deepbayes
#include "cv_error_function.hpp"

// [[Rcpp::export]]
List optimize(NumericMatrix xPoints_mat,
              int k,
              int budget,
              std::vector<std::string> param_prefixes,
              std::string solver_dir,
              std::string log_dir) {

    // Default parameters.  See the following for possible values:
    // https://github.com/rmcantin/bayesopt/blob/master/src/parameters.cpp
    // TODO: take parameter values as input to function
    bayesopt::Parameters par;
    par.n_iterations = budget;

    // nPoints is the number of possible values, and n is the dimension of
    // the hyperparameter matrix.
    const int nPoints = xPoints_mat.ncol();
    const int n = xPoints_mat.nrow();

    // Flatten out our hyperparameter matrix into a vector in order to input
    // to the C API of BayesOpt.
    const int size = n * nPoints;
    double xPointsArray[size];
    for (int i = 0; i < nPoints; ++i) {
        NumericVector col = xPoints_mat(_, i);
        for (int j = 0; j < n; ++j) {
            xPointsArray[i * n + j] = col[j];
        }
    }

    std::cout << "xPointsArray: ";
    for (int i = 0; i < size; ++i) {
        std::cout << xPointsArray[i] << ", ";
    }
    std::cout << std::endl;

    // 128 used as standard size value
    // (see Ruben's explanation in BayesOpt Github)
    double x[128], fmin[128];

    // Fill out the struct
    user_function_data ui_data;
    ui_data.k = k;
    ui_data.param_prefixes = param_prefixes;
    ui_data.solver_dir = solver_dir;
    ui_data.log_dir = log_dir;

    bayes_optimization_disc(n,
                            &cv_error_function,
                            &ui_data,
                            xPointsArray,
                            nPoints,
                            x,
                            fmin,
                            par.generate_bopt_params());

    std::cout << "Final result C: ["<< n << "](";
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << ", ";
    }
    std::cout << ")" << " | Value:" << fmin[0] << std::endl;

    NumericVector x_opt = NumericVector::create(*x);
    NumericVector f = NumericVector::create(fmin[0]);
    List z = List::create(x_opt, f);
    return(z);
}