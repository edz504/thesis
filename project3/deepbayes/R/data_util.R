#####
# Contains helper functions for simulating, splitting, and preparing data for
# use in Caffe.
#####

###
# Given an input dimension, an output dimension, simulates N data from a neural
# network with a single hidden layer.  Set a seed for reproducability if given,
# and add noise if specified.
###
simulate_fc_data <- function(x_dim, y_dim, num_nodes, N, seed, noise) {
    
    if (!is.null(seed)) {
        set.seed(seed)
    }

    # We randomly sample Unif(-1, 1) to create the weights into the hidden
    # layer and into the output layer.
    W <- matrix(runif(x_dim * num_nodes, -1, 1), num_nodes, x_dim)
    W2 <- matrix(runif(y_dim * num_nodes, -1, 1), y_dim, num_nodes)

    # We create our dataset by sampling from a normal distribution N times.
    X <- matrix(rnorm(x_dim * N), x_dim, N)

    # input -> FC -> sigmoid activation
    h1 <- 1 / (1 + exp(-(W %*% X)))

    # -> output (note: output is not activated)
    y <- W2 %*% h1
    # y <- y + matrix(rnorm(y_dim * N, sd=1), y_dim, N)

    # Note that X and y both have data column-wise in the data generation,
    # and we transpose before returning them.
    return(list(X=t(X), y=t(y)))
}

###
# Similar network structure as above, but we consider 1-D binary
# output.
###
simulate_fc_binary <- function(x_dim, num_nodes, N, seed, noise) {
    if (!is.null(seed)) {
        set.seed(seed)
    }

    # We randomly sample Unif(-1, 1) to create the weights into the hidden
    # layer and into the output layer.
    W <- matrix(runif(x_dim * num_nodes, -1, 1), num_nodes, x_dim)
    W2 <- matrix(runif(num_nodes, -1, 1), 1, num_nodes)

    # We create our dataset by sampling from a normal distribution N times.
    X <- matrix(rnorm(x_dim * N), x_dim, N)

    # input -> FC -> sigmoid activation
    h1 <- 1 / (1 + exp(-(W %*% X)))

    # -> output (note: output is not activated)
    h2 <- 1 / (1 + exp(-(W2 %*% h1)))

    y <- round(h2)

    # Note that X and y both have data column-wise in the data generation,
    # and we transpose before returning them.
    return(list(X=t(X), y=t(y)))
}

###
# Given a vector containing layer sizes and data size N, simulates N data from
# a neural network.  For example, if [2, 5, 10, 5, 3] is provided, simulate
# data with input dimension 2, 3 hidden layers with 5, 10, and 5 nodes
# respectively, and output dimension 3.  Set a seed for reproducability if
# given.
###
simulate_multiple_fc_data <- function(network_vec, N, seed) {
    
    if (!is.null(seed)) {
        set.seed(seed)
    }

    if (length(network_vec) < 2) {
        stop("Need at least an input and output layer")
    }

    num_layers <- length(network_vec)
    x_dim <- network_vec[1]
    y_dim <- network_vec[num_layers]
    # We create our dataset by sampling from a normal distribution N times.
    X <- matrix(rnorm(x_dim * N), x_dim, N)
    h <- X

    # We can justify using a loop here instead of vectorization because
    # of the magnitude of the depth of the network.
    for (i in 1:(length(network_vec) - 2)) {
        in_dim <- network_vec[i]
        out_dim <- network_vec[i + 1]
        W <- matrix(runif(in_dim * out_dim, -1, 1), out_dim, in_dim)
        h <- 1 / (1 + exp(-W %*% h))
    }

    # -> output (note: output is not activated)
    W <- matrix(runif(out_dim * y_dim, -1, 1), y_dim, out_dim)
    y <- W %*% h

    # Note that X and y both have data column-wise in the data generation,
    # and we transpose before returning them.
    return(list(X=t(X), y=t(y)))
}


###
# Given k, a number of folds, input and output matrix-like objects X, y,
# split the data into k folds.  Data is saved in HDF5 format for use in Caffe,
# with each fold's training and validation sets represented as
# <prefix>_train<i>.h5 and <prefix>_valid<i>.h5.  For example, a size-100
# dataset with 5 folds (call it "fc_example") would have 80 data in
# fc_example_train1.h5 and 20 in fc_example_valid1.h5. Data is assumed to be
# shuffled (this can easily be changed later), and is saved to
# deepbayes/datasets.
###
kfold_split <- function(k, X, y, prefix) {

    N <- nrow(X)
    fold_size <- N / k

    for (i in 1:k) {
        start <- ((i - 1) * fold_size) + 1
        end <- start + fold_size - 1
        valid_ind <- start:end

        train_X <- X[-valid_ind, ]  # Add noise to training set for overfitting
        # train_y <- y[-valid_ind, ] + matrix(rnorm(
        #                                     ncol(y) * (N - fold_size),
        #                                     mean=0, sd=1),
        #                                     (N - fold_size), ncol(y))
        train_y <- y[-valid_ind, ]
        valid_X <- X[valid_ind, ]
        valid_y <- y[valid_ind, ]

        dataset_path <- system.file('datasets', package='deepbayes')

        train_file <- h5file(paste(dataset_path, "/", prefix, "_train", i,
                                   ".h5", sep=''))
        train_file["data"] <- train_X
        train_file["label"] <- train_y
        h5close(train_file)

        # We also have to create a text file that holds the hdf5 name
        train_txtfile_name <- paste(dataset_path, "/", prefix, "_train", i,
                                    ".txt", sep='')
        file.create(train_txtfile_name)
        train_textfile <- file(train_txtfile_name)
        writeLines(c(paste(dataset_path, "/", prefix, "_train", i, ".h5", sep='')),
                   train_textfile)
        close(train_textfile)

        valid_file <- h5file(paste(dataset_path, "/", prefix, "_valid", i, ".h5",
                                   sep=''))
        valid_file["data"] <- valid_X
        valid_file["label"] <- valid_y
        h5close(valid_file)
        valid_txtfile_name <- paste(dataset_path, "/", prefix, "_valid", i, ".txt",
                                    sep='')
        file.create(valid_txtfile_name)
        valid_textfile <- file(valid_txtfile_name)
        writeLines(c(paste(dataset_path, "/", prefix, "_valid", i, ".h5",
                           sep='')),
                   valid_textfile)
        close(valid_textfile)
    }
}