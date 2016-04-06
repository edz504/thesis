#####
# This example function is written to demonstrate interfacing BayesOpt as a
# hyperparameter tuning framework for (deep) neural networks with Caffe.
# We take the set of nets where we have a 3-d input layer, a fixed sequence
# of multi-dimensional hidden layers and a 2-d output layer.  We seek to
# tune the number of training iterations (max_iter in Caffe).
#####
max_iter_example <- function() {

    N <- 10000
    k <- 2
    max_iter_vals <- seq(2500, 100000, by=2500)

    cat("Simulating data...\n")
    sim <- simulate_multiple_fc_data(
      c(3, 5, 10, 5, 2),
        N, NULL)
    X <- sim[[1]]
    y <- sim[[2]]

    cat("Splitting into folds...\n")
    prefix <- "max_iter"
    kfold_split(k, X, y, prefix)                              # See data_util.R

    cat("Creating net and solver protos...\n")
    create_protos_max_iter(N, k, max_iter_vals, prefix)       # See below

    # Call the method from Rcpp that optimizes our validation error
    # function.  Note that we reshape into a 1 row matrix for generality.
    t1 <- proc.time()
    budget <- 5                                               # Set budget
    opt_out <- optimize(matrix(max_iter_vals,
                               1, length(max_iter_vals)),
                        k,
                        budget,
                        prefix,
                        solver_path,
                        log_path)
    T1 <- proc.time()

    # Call the brute force optimizer to check.  Note that if max_iter_vals
    # is very large, this will take a long time.
    t2 <- proc.time()
    brute_out <- brute_force(matrix(max_iter_vals,
                                    1, length(max_iter_vals)),
                             k,
                             prefix,
                             solver_path,
                             log_path)
    T2 <- proc.time()

    cat("bayesopt:\n")
    cat(T1 - t1, "\n")
    cat("brute force:\n")
    cat(T2 - t2, "\n")
}


# This function takes in a dataset size N, number of folds k, range of
# max_iter values, and prefix name.  Creates the net and solver proto files
# corresponding to this example for tuning max_iter.  The template is a 
# 3->3->5->10->5->2 network, made large to demonstrate overfitting.
create_protos_max_iter <- function(N, k, max_iter_vals, prefix) {

    # Create a net .prototxt for each fold.  Note here that the hyperparameter
    # we tune is a value specified in the solver, not in the network structure.
    # Therefore, we only need 5 net protos (1 for each fold).

    # Retrieve the template prototxt
    net_template_file <- file(paste(net_path,
                                    '/max_iter_train_valid_template.prototxt',
                                    sep=''))
    for (i in 1:k) {
        template_lines <- readLines(net_template_file)
        # Change the name of the net
        template_lines[1] <- paste('name: "fold', i, '"', sep='')
        # Change the training dataset
        template_lines[8] <- paste('    source: "',
                                   dataset_path,
                                   '/',
                                   prefix,
                                   '_train',
                                   i, '.txt"', sep='')
        # Change the validation dataset
        template_lines[21] <- paste('    source: "',
                                   dataset_path,
                                   '/',
                                   prefix,
                                   '_valid',
                                   i, '.txt"', sep='')
        this_fold_net_filename <- paste(net_path,
                                        '/fold', i,
                                        '_train_valid.prototxt', sep='')
        file.create(this_fold_net_filename)
        this_fold_net_file <- file(this_fold_net_filename)
        writeLines(template_lines,
                  this_fold_net_file)
        close(this_fold_net_file)
    }

    # Create k solver .prototxt files for each value of the hyperparameter
    # that we consider.  This means we create k * n solver .prototxt files,
    # where n is the number of different max_iter values in consideration.
    solver_template_file <- file(paste(solver_path,
                                       '/max_iter_n_fold_k_solver.prototxt',
                                       sep=''))
    for (max_iter in max_iter_vals) {
        # Get rid of scientific notation for formatting
        max_iter <- as.integer(max_iter)
        for (i in 1:k) {
            template_lines <- readLines(solver_template_file)
            # Change the net referred to be the one pointing at this fold
            template_lines[1] <- paste('net: "',
                                       net_path,
                                       '/fold',
                                       i, '_train_valid.prototxt"', sep='')

            # Change the test_iter (test_iter * batch_size = test_size)
            test_iter <- (N / k) / 100
            template_lines[2] <- paste('test_iter: ', test_iter, sep='')

            # Change the max_iter
            template_lines[11] <- paste('max_iter: ', max_iter, sep='')

            # Change the snapshot prefix (note that "iter" is automatically
            # added to the saved model filename)
            template_lines[13] <- paste('snapshot_prefix: "',
                                        model_path,
                                        '/fold',
                                        i, '"', sep='')

            # Change the solver to only test, display, and snapshot at the final
            # iteration max_iter
            template_lines[3] <- paste('test_interval: ', max_iter, sep='')
            template_lines[10] <- paste('display: ', max_iter, sep='')
            template_lines[12] <- paste('snapshot: ', max_iter, sep='')
            
            this_solver_filename <- paste(solver_path,
                                          '/max_iter_',
                                          max_iter, # sci notation
                                          '_fold_', i,
                                          "_solver.prototxt",
                                          sep='')
            file.create(this_solver_filename)
            this_solver_file <- file(this_solver_filename)
            writeLines(template_lines,
                       this_solver_file)
            close(this_solver_file)
        }
    }
}