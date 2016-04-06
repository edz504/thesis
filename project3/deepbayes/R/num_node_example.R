#####
# This example function is written to demonstrate interfacing BayesOpt as a
# hyperparameter tuning framework for (deep) neural networks with Caffe.
# We take the simple set of nets where we have a 3-d input layer, a hidden layer
# with n nodes where n is the hyperparameter to be tuned, and a 2-d output layer.
#####
num_node_example <- function() {

    N <- 10000
    k <- 5
    in_dim <- 3
    out_dim <- 2
    num_nodes <- 10
    num_node_vals <- seq(1, 1000)
    prefix <- "num_node"

    cat("Simulating data...\n")
    sim <- simulate_fc_data(in_dim, out_dim, num_nodes,       # See data_util.R
                            N, 3333, TRUE)         
    X <- sim[[1]]
    y <- sim[[2]]

    cat("Splitting into folds...\n")
    kfold_split(k, X, y, prefix)                              # See data_util.R

    cat("Creating net and solver protos...\n")
    create_protos_num_node(N, k, num_node_vals, prefix)       # See below
  
    t <- proc.time()
    budget <- 25
    opt_out <- optimize(matrix(num_node_vals,
                               1, length(num_node_vals)),
                        k,
                        budget,
                        prefix,
                        solver_path,
                        log_path)
    T <- proc.time()
    t2 <- proc.time()
    brute_out <- brute_force(matrix(num_node_vals,
                                    1, length(num_node_vals)),
                             k,
                             prefix,
                             solver_path,
                             log_path)
    T2 <- proc.time()

    cat("bayesopt:\n")
    cat(T - t, "\n")
    cat("brute force:\n")
    cat(T2 - t2, "\n")
}


# This function takes in a dataset size N, number of folds k, range of
# num_nodes values, and prefix name.  Creates the net and solver proto files
# corresponding to this example for tuning num_nodes.
create_protos_num_node <- function(N, k, num_node_vals, prefix) {

    # Create a net .prototxt for each fold.  Note here that the hyperparameter
    # we tune is embedded in the network structure; it is not a value specified
    # in the solver.  Therefore, we need k * m net protos (1 for each fold),
    # where m is the number of different num_node_vals.

    # Retrieve the template prototxt
    net_template_file <- file(paste(net_path,
                                    '/num_node_train_valid_template.prototxt',
                                    sep=''))
    for (i in 1:k) {
        for (n in num_node_vals) {
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
            # Change the num_output (number of nodes) in the hidden layer
            template_lines[34] <- paste('    num_output: ', n, sep='')

            # Name the file
            this_fold_net_filename <- paste(net_path,
                                            '/fold', i,
                                            '_num_node_', n,
                                            '_train_valid.prototxt', sep='')
            file.create(this_fold_net_filename)
            this_fold_net_file <- file(this_fold_net_filename)
            writeLines(template_lines,
                      this_fold_net_file)
            close(this_fold_net_file)
        }
    }

    # Create k * m solver .prototxt files, one for each net .prototxt.
    # Note that the only thing changing from solver to solver is the net
    # referred to, no other hyperparameters within the solver are changed.
    solver_template_file <- file(paste(solver_path,
                                       '/num_node_m_fold_k_solver.prototxt',
                                       sep=''))
    for (n in num_node_vals) {
        for (i in 1:k) {
            template_lines <- readLines(solver_template_file)
            # Change the net referred to be the one pointing at this fold
            template_lines[1] <- paste('net: "',
                                       net_path,
                                       '/fold', i,
                                       '_num_node_', n,
                                       '_train_valid.prototxt"', sep='')

            # Change the test_iter (test_iter * batch_size = test_size)
            test_iter <- (N / k) / 100
            template_lines[2] <- paste('test_iter: ', test_iter, sep='')

            # Change the snapshot prefix
            template_lines[13] <- paste('snapshot_prefix: "',
                                        model_path,
                                        '/fold', i,
                                        '_num_node_', n, '"', sep='')
            
            this_solver_filename <- paste(solver_path,
                                          '/num_node_',
                                          n,
                                          '_fold_',
                                          i,
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