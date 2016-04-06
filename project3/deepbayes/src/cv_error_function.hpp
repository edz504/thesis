#ifndef CV_ERROR_FUNCTION_H
#define CV_ERROR_FUNCTION_H

// struct for user_function_data
typedef struct user_function_data{
    int k;
    std::vector<std::string> param_prefixes;
    std::string solver_dir, log_dir;
} user_function_data;

double cv_error_function(unsigned int n, const double *x,
                         double *gradient, /* NULL if not needed */
                         void *func_data);
#endif
