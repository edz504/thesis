// Caffe
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

// Parsing
#include <fstream>
#include <string>
#include <stdlib.h>
#include <sstream>

// deepbayes
#include "cv_error_function.hpp"

using caffe::Caffe;

// Note: currently 1D in x
double cv_error_function(unsigned int n, const double *x,
            double *gradient, /* NULL if not needed */
            void *func_data) {

    // Extract void pointer data
    const user_function_data* ui_data = (struct user_function_data*)func_data;

    // Read in the solver prototxt corresponding to the provided value x
    // and solve
    caffe::SolverParameter solverParams;
    caffe::Solver<float>* solver;
    double total_loss = 0;
    for (int i = 0; i < ui_data->k; ++i) {
        std::string solver_proto = ui_data->solver_dir + "/";
        // std::vector<double> input;
        // input.assign(*x, *x + ui_data->param_prefixes.size());
        for (unsigned int j = 0; j < n; ++j) {
            std::ostringstream input_stm;
            input_stm << x[j];
            std::string input_str = input_stm.str();

            solver_proto = solver_proto + 
                           ui_data->param_prefixes[j] + "_" +
                           input_str + "_";
        }

        std::ostringstream fold_stm;
        fold_stm << i + 1;
        std::string fold_string = fold_stm.str();

        solver_proto = solver_proto + "fold_" + fold_string +
                       "_solver.prototxt";

        caffe::ReadProtoFromTextFileOrDie(solver_proto, &solverParams);
        solver = new caffe::SGDSolver<float>(solverParams);
        solver->Solve();

        // Get 2nd to last line for loss
        // TODO: fix glog symlink to not be UNKNOWN.INFO
        std::string loss_path = ui_data->log_dir + "/UNKNOWN.INFO";
        std::ifstream file(loss_path.c_str());
        std::vector<std::string> lines;
        std::string str;
        while (std::getline(file, str))
        {
            lines.push_back(str);
        }
        // Parse loss from line
        std::string final_loss_line = lines[lines.size() - 2];
        int start = final_loss_line.find("loss = ") + 7;
        int end = final_loss_line.find(" (");
        std::string final_loss_str = final_loss_line.substr(start,
                                                            end - start);
        double final_loss = ::atof(final_loss_str.c_str());
        total_loss += final_loss;
    }

    return(total_loss / ui_data->k);
}
