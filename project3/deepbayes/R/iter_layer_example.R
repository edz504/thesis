#####
# This example function is written to demonstrate interfacing BayesOpt as a
# hyperparameter tuning framework for (deep) neural networks with Caffe.
# We want to show the potential for tuning multiple hyperparameters at the
# same time.  In this case, we consider both max_iter (embedded in the solver)
# and number of hidden layers (each with a constant number of nodes).
#####
iter_layer_example <- function() {

    N <- 10000
    k <- 2
    iter_vals <- c(100, 500, 1000, 5000, 10000)
    layer_vals <- c(5, 10, 25, 50, 100, 500)

    cat("Simulating data...\n")
    sim <- simulate_multiple_fc_data(
      c(3, 10, 10, 10, 2),
        N, NULL)
    X <- sim[[1]]
    y <- sim[[2]]

    cat("Splitting into folds...\n")
    prefix <- "iter_layer"
    kfold_split(k, X, y, prefix)                              

    cat("Creating net and solver protos...\n")
    create_protos_iter_layer(N, k, iter_vals, layer_vals, prefix)

    # This creates a matrix with each length-2 column representing a
    # combination of an iter_val and a layer_val
    iter_layer_vals <- t(expand.grid(iter_vals, layer_vals))
    t1 <- proc.time()
    budget <- 5                                               # Set budget
    opt_out <- optimize(iter_layer_vals,
                        k,
                        budget,
                        strsplit(prefix, '_')[[1]], # Split prefix into str vec
                        solver_path,
                        log_path)
    T1 <- proc.time()

    # Call the brute force optimizer to check.  Note that if max_iter_vals
    # is very large, this will take a long time.
    t2 <- proc.time()
    brute_out <- brute_force(iter_layer_vals,
                             k,
                             strsplit(prefix, '_')[[1]],
                             solver_path,
                             log_path)
    T2 <- proc.time()

    cat("bayesopt:\n")
    cat(T1 - t1, "\n")
    cat("brute force:\n")
    cat(T2 - t2, "\n")
}

# In this implementation, we create our protos from hardcoded text rather than
# a template, because we need to do more than simply change a few values.
create_protos_iter_layer <- function(N, k, iter_vals, layer_vals, prefix) {
    for (num_layers in layer_vals) {
        for (i in 1:k) {
            # Train and test data layers are the same.  Also add the first
            # FC layer, because its "bottom" is "data", not "ip<j-1>".
            net_txt <- paste('name: "iter_layer"
layer {
    name: "data"
    type: "HDF5Data"
    top: "data"
    top: "label"
    hdf5_data_param {
        source: "', dataset_path,
                '/',
                prefix,
                '_train',
                i, '.txt"
        batch_size: 100
    }
    include: {
        phase: TRAIN
    }
}
layer {
    name: "data"
    type: "HDF5Data"
    top: "data"
    top: "label"
    hdf5_data_param {
        source: "', dataset_path,
                '/',
                prefix,
                '_valid',
                i, '.txt"
        batch_size: 100
    }
    include: {
        phase: TEST
    }
}
layer {
    name: "ip1"
    type: "InnerProduct"
    bottom: "data"
    top: "ip1"
    inner_product_param {
        num_output: 10
        weight_filler {
            type: "uniform"
            min: -1
            max: 1
        }
        bias_term: false
    }
}
', sep='')
            # Add (num_layers - 1) Sigmoid and FC layers
            if (num_layers > 1) {
                for (j in 1:(num_layers - 1)) {
                    net_txt <- paste(net_txt,
'layer {
    name: "sigmoid', j,'"
    type: "Sigmoid"
    bottom: "ip', j,'"
    top: "sigmoid', j,'"
}
layer {
    name: "ip', j + 1,'"
    type: "InnerProduct"
    bottom: "sigmoid', j,'"
    top: "ip', j + 1,'"
    inner_product_param {
        num_output: 10
        weight_filler {
            type: "uniform"
            min: -1
            max: 1
        }
        bias_term: false
    }
}
', sep='')
                }
            }

            # Add the last Sigmoid, FC layer (note: last FC layer should
            # have num_output 3), and Loss layer.
            net_txt <- paste(net_txt,
'layer {
    name: "sigmoid', num_layers,'"
    type: "Sigmoid"
    bottom: "ip', num_layers,'"
    top: "sigmoid', num_layers,'"
}
layer {
    name: "ip', num_layers + 1,'"
    type: "InnerProduct"
    bottom: "sigmoid', num_layers,'"
    top: "ip', num_layers + 1,'"
    inner_product_param {
        num_output: 2
        weight_filler {
            type: "uniform"
            min: -1
            max: 1
        }
        bias_term: false
    }
}
layer {
    name: "loss"
    type: "EuclideanLoss"
    bottom: "ip', num_layers + 1,'"
    bottom: "label"
    top: "loss"
}
', sep='')
            this_fold_net_filename <- paste(net_path,
                                            '/num_layers_', num_layers,
                                            '_fold', i,
                                            '_train_valid.prototxt', sep='')
            file.create(this_fold_net_filename)
            this_fold_net_file <- file(this_fold_net_filename)
            write(net_txt,
                      this_fold_net_file)
            close(this_fold_net_file)


            # Create the solver files.  Note that the solver filename should
            # parallel the prefix, so prefix "iter_layer" corresponds to
            # solvers of the form "iter_X_layer_Y_fold_k_solver.prototxt".
            solver_template_file <- file(paste(solver_path,
                                       '/max_iter_n_fold_k_solver.prototxt',
                                       sep=''))
            for (max_iter in iter_vals) {
                # Get rid of scientific notation for formatting
                max_iter <- as.integer(max_iter)
                template_lines <- readLines(solver_template_file)
                # Change the net referred to be the one pointing at this fold
                template_lines[1] <- paste('net: "',
                                           net_path,
                                           '/num_layers_', num_layers,
                                           '_fold', i,
                                           '_train_valid.prototxt"', sep='')

                # Change the test_iter (test_iter * batch_size = test_size)
                test_iter <- (N / k) / 100
                template_lines[2] <- paste('test_iter: ', test_iter, sep='')

                # Change the max_iter
                template_lines[11] <- paste('max_iter: ', max_iter, sep='')

                # Change the snapshot prefix (note that "iter" is automatically
                # added to the saved model filename)
                template_lines[13] <- paste('snapshot_prefix: "',
                                            model_path,
                                            '/num_layers_', num_layers,
                                            '_fold', i,
                                            '"', sep='')

                # Change the solver to only test, display, and snapshot at the final
                # iteration max_iter
                template_lines[3] <- paste('test_interval: ', max_iter, sep='')
                template_lines[10] <- paste('display: ', max_iter, sep='')
                template_lines[12] <- paste('snapshot: ', max_iter, sep='')
                
                this_solver_filename <- paste(solver_path,
                                              '/iter_', max_iter,
                                              '_layer_', num_layers,
                                              '_fold_', i,
                                              "_solver.prototxt",
                                              sep='')
                file.create(this_solver_filename)
                this_solver_file <- file(this_solver_filename)
                writeLines(template_lines,
                           this_solver_file)
                close(this_solver_file)
            }
            close(solver_template_file)

        }
    }

}