.onLoad <- function(libname, pkgname) {
    # Make paths to files (net and solver protos, dataset, loss logs)
    # global
    dataset_path <<- system.file('datasets', package='deepbayes')
    net_path <<- system.file('nets', package='deepbayes')
    solver_path <<- system.file('solvers', package='deepbayes')
    model_path <<- system.file('models', package='deepbayes')
    log_path <<- system.file('logs', package='deepbayes')

    # Call the C++ exported method for glog flag setting
    setup(paste(log_path, '/', sep=''))
}