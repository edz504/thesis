library(ggplot2)
library(reshape2)

x <- seq(-3, 3, by=0.01)
relu <- function(x) {
    return(sapply(x, FUN=max, 0))
}
sigmoid <- function(x) {
    return(1 / (1 + exp(-x)))
}

relu.vec <- relu(x)
# sigmoid.vec <- sigmoid(x)

# df <- data.frame(x, relu.vec, sigmoid.vec)
# df$beta_01 <- relu.vec - 0.1 * sigmoid.vec
# df$beta_05 <- relu.vec - 0.5 * sigmoid.vec
# df$beta_1 <- relu.vec - sigmoid.vec
# df$beta_2 <- relu.vec - 2 * sigmoid.vec
# df$beta01 <- relu.vec + 0.1 * sigmoid.vec
# df$beta05 <- relu.vec + 0.5 * sigmoid.vec
# df$beta1 <- relu.vec + sigmoid.vec
# df$beta2 <- relu.vec + 2 * sigmoid.vec

# plot.df <- melt(df, id='x')
# ggplot(plot.df, aes(x=x, y=value, colour=variable)) +
#     geom_line()

sin.vec <- sin(x)
df <- data.frame(x, relu.vec, sin.vec)
df$beta_01 <- relu.vec - 0.1 * sin.vec
df$beta_05 <- relu.vec - 0.5 * sin.vec
df$beta_1 <- relu.vec - sin.vec
df$beta_2 <- relu.vec - 2 * sin.vec
df$beta01 <- relu.vec + 0.1 * sin.vec
df$beta05 <- relu.vec + 0.5 * sin.vec
df$beta1 <- relu.vec + sin.vec
df$beta2 <- relu.vec + 2 * sin.vec

plot.df <- melt(df, id='x')
ggplot(plot.df, aes(x=x, y=value, colour=variable)) +
    geom_line()