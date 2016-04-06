// BayesOpt
#include <bayesopt/bayesopt.h>
#include <bayesopt/bayesopt.hpp>

// Caffe
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

// Rcpp
#include <Rcpp.h>
using namespace Rcpp;

// other
#include <limits>
#include <string>
#include <stdlib.h>

// deepbayes
#include "cv_error_function.hpp"

using caffe::Caffe;

// [[Rcpp::export]]
List max_iter_optimize(NumericMatrix xPoints_mat,
                       int k,
                       int budget,
                       std::string solver_dir,
                       std::string log_dir) {

    // Default parameters.  See the following for possible values:
    // https://github.com/rmcantin/bayesopt/blob/master/src/parameters.cpp
    // TODO(Eddie): take parameter values as input to function
    bayesopt::Parameters par;
    par.n_iterations = budget;

    // Note that in the max_iter case, we only have 1 hyperparameter to tune,
    // so our xPoints_mat is really a vector.
    const int nPoints = xPoints_mat.ncol();
    const int n = xPoints_mat.nrow();

    // Use ublas vectors to flatten out our hyperparameter matrix (which,
    // again, is really just a vector -- we provide this flattening for
    // example).
    const int size = n * nPoints;
    double xPointsArray[size];
    for (int i = 0; i < n; ++i) {
        NumericVector row = xPoints_mat(i, _);
        for (int j = 0; j < nPoints; ++j) {
            xPointsArray[i * n + j] = row[j];
        }
    }

    // Waiting on Ruben to clarify why 128 is used
    double x[128], fmin[128];

    // Fill out the struct
    user_function_data ui_data;
    ui_data.k = k;
    ui_data.param_prefix = "max_iter";
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
    NumericVector x_opt = NumericVector::create(1, 2);
    NumericVector f = NumericVector::create(0.69);
    List z = List::create(x_opt, f);
    return(z);
}

// [[Rcpp::export]]
void max_iter_brute_force(NumericMatrix xPoints_mat,
                          int k,
                          std::string solver_dir,
                          std::string log_dir) {

    const int nPoints = xPoints_mat.ncol();
    const int n = xPoints_mat.nrow();
    user_function_data ui_data;
    ui_data.k = k;
    ui_data.param_prefix = "max_iter";
    ui_data.solver_dir = solver_dir;
    ui_data.log_dir = log_dir;

    NumericVector row = xPoints_mat(0, _);
    double min_f = std::numeric_limits<double>::infinity();
    int min_x = 0;
    double this_f;
    for (int j = 0; j < nPoints; ++j) {
        this_f = cv_error_function(n, &row[j], NULL, &ui_data);
        std::cout << "x = " << row[j] << ": f = " << this_f << std::endl;
        if (this_f < min_f) {
            min_x = row[j];
            min_f = this_f;
        }
    }

    std::cout << "min x = " << min_x <<": achieves f = " << min_f << std::endl;

}