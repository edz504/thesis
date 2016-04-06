// BayesOpt
#include <boost/numeric/ublas/matrix_proxy.hpp>  // For row()
#include <bayesopt/bayesopt.h>
#include <bayesopt/bayesopt.hpp>
#include <specialtypes.hpp>

// Caffe
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

// Parsing
#include <fstream>
#include <string>
#include <stdlib.h>

// Rcpp
#include <Rcpp.h>
using namespace Rcpp;

// other
#include <sstream>
#include <limits>

// struct for user_function_data
typedef struct num_node_data{
    int k;
    std::string solver_dir, log_dir;
} num_node_data;

using caffe::Caffe;

double num_node_cv_error(unsigned int n, const double *x,
            double *gradient, /* NULL if not needed */
            void *func_data) {

    // Extract void pointer data
    const num_node_data* nn_data = (struct num_node_data*)func_data;

    // Read in the solver prototxt corresponding to the provided value x
    // and solve
    caffe::SolverParameter solverParams;
    caffe::Solver<float>* solver;
    double total_loss = 0;
    for (int i = 0; i < nn_data->k; ++i) {
        int num_node_int = *x;
        std::ostringstream stm;
        stm << num_node_int;
        std::string num_node_string = stm.str();
        std::ostringstream stm2;
        stm2 << i + 1;
        std::string fold_string = stm2.str();

        std::string solver_proto = nn_data->solver_dir +
                                   "/num_node_" +
                                   num_node_string +
                                   "_fold_" +
                                   fold_string +
                                   "_solver.prototxt";
        caffe::ReadProtoFromTextFileOrDie(solver_proto, &solverParams);
        solver = new caffe::SGDSolver<float>(solverParams);
        solver->Solve();

        // Get 2nd to last line for loss
        std::string loss_path = nn_data->log_dir + "/UNKNOWN.INFO";
        std::ifstream file(loss_path.c_str());
        std::vector<std::string> lines;
        std::string str;
        while (std::getline(file, str))
        {
            lines.push_back(str);
        }
        // Parse loss from line
        std::string final_loss_line = lines[lines.size() - 2];
        // std::cout << final_loss_line << std::endl;
        int start = final_loss_line.find("loss = ") + 7;
        int end = final_loss_line.find(" (");
        std::string final_loss_str = final_loss_line.substr(start,
                                                            end - start);
        double final_loss = ::atof(final_loss_str.c_str());
        total_loss += final_loss;
    }

    return(total_loss / nn_data->k);
}

// [[Rcpp::export]]
List num_node_optimize(NumericMatrix xPoints_mat,
                       int k,
                       int budget,
                       std::string solver_dir,
                       std::string log_dir) {

    // Default parameters.  See the following for possible values:
    // https://github.com/rmcantin/bayesopt/blob/master/src/parameters.cpp
    // TODO(Eddie): take parameter values as input to function
    bayesopt::Parameters par;
    par.n_iterations = budget;

    // Note that in the num_node case, we only have 1 hyperparameter to tune,
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
    num_node_data nn_data;
    nn_data.k = k;
    nn_data.solver_dir = solver_dir;
    nn_data.log_dir = log_dir;

    bayes_optimization_disc(n,
                            &num_node_cv_error,
                            &nn_data,
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
void num_node_brute_force(NumericMatrix xPoints_mat,
                          int k,
                          std::string solver_dir,
                          std::string log_dir) {

    const int nPoints = xPoints_mat.ncol();
    const int n = xPoints_mat.nrow();
    num_node_data nn_data;
    nn_data.k = k;
    nn_data.solver_dir = solver_dir;
    nn_data.log_dir = log_dir;

    NumericVector row = xPoints_mat(0, _);
    double min_f = std::numeric_limits<double>::infinity();
    int min_x = 0;
    double this_f;
    for (int j = 0; j < nPoints; ++j) {
        this_f = num_node_cv_error(n, &row[j], NULL, &nn_data);
        // std::cout << "x = " << row[j] << ": f = " << this_f << std::endl;
        if (this_f < min_f) {
            min_x = row[j];
            min_f = this_f;
        }
    }

    std::cout << "min x = " << min_x <<": achieves f = " << min_f << std::endl;

}