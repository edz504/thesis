library(ggplot2)
library(reshape2)
library(dplyr)

parametric_init_losses <- read.csv('parametric_init_losses.csv')
parametric_init_losses_2 <- read.csv('parametric_init_losses_20k-50k.csv')

plot.df <- melt(rbind(parametric_init_losses,
                      parametric_init_losses_2),
                id='iteration')
plot.df$iteration <- plot.df$iteration * 100 # output every 100 iter

# ggplot(plot.df, aes(x=iteration, y=value)) +
#     geom_line(aes(colour=variable)) +
#     ggtitle('LSTM Parametric Initialization Loss during Training')
# ggsave('parametric_initialization_50k_iter.png',
#        width=10, height=6)


## colours help
gg_color_hue <- function(n) {
    hues <- seq(15, 375, length=n+1)
    hcl(h=hues, l=65, c=100)[1:n]
}


# # 20k->20k
parametric_init <- read.csv('parametric_init_losses.csv')
parametric_init_2 <- read.csv('parametric_init_losses_20k-50k.csv')
parametric_init_df <- rbind(parametric_init, parametric_init_2[1:200, ])

# np_20 <- read.csv('lstm_np_02_loss.csv')

# # First plot 20k init -> 20k np
# plot_df <- data.frame(init_iteration=parametric_init$iteration,
#                       init_test_loss=parametric_init$testing_loss,
#                       np_iteration=np_20$iteration,
#                       np_test_loss=np_20$testing_loss)
# plot_df$init_iteration <- plot_df$init_iteration * 100
# plot_df$np_iteration <- plot_df$np_iteration * 100

# ggplot(plot_df) +
#     geom_line(aes(x=init_iteration, y=init_test_loss),
#               colour=gg_color_hue(2)[1]) +
#     geom_line(aes(x=np_iteration, y=np_test_loss),
#               colour=gg_color_hue(2)[2]) +
#     ggtitle('Nonparametric Testing Loss, Parametric Initialization') +
#     xlab('Iteration') +
#     ylab('Testing Loss')
# ggsave(file='np20_init20.png', width=10, height=6)

# # Also plot 20k continuation from parametric
# plot_df <- data.frame(init_iteration=parametric_init_df$iteration,
#                       init_test_loss=parametric_init_df$testing_loss,
#                       np_iteration=np_20$iteration,
#                       np_test_loss=np_20$testing_loss)
# plot_df$init_iteration <- plot_df$init_iteration * 100
# plot_df$np_iteration <- plot_df$np_iteration * 100

# ggplot(plot_df) +
#     geom_line(aes(x=init_iteration, y=init_test_loss),
#               colour=gg_color_hue(2)[1]) +
#     geom_line(aes(x=np_iteration, y=np_test_loss),
#               colour=gg_color_hue(2)[2]) +
#     ggtitle('Nonparametric Testing Loss, Parametric Initialization and Continuation') +
#     xlab('Iteration') +
#     ylab('Testing Loss')
# ggsave(file='np20_init20_cont20.png', width=10, height=6)


# Now 50k parametric, 50k nonparametric
parametric_init_df <- rbind(parametric_init,
                            parametric_init_2)
np_sigmoid_50 <- read.csv('lstm_np_01_loss.csv')
plot_df <- data.frame(init_iteration=parametric_init_df$iteration,
                      init_test_loss=parametric_init_df$testing_loss,
                      np_iteration=np_sigmoid_50$iteration,
                      np_test_loss=np_sigmoid_50$testing_loss)
plot_df$init_iteration <- plot_df$init_iteration * 100
plot_df$np_iteration <- plot_df$np_iteration * 100

ggplot(plot_df) +
    geom_line(aes(x=init_iteration, y=init_test_loss),
              colour=gg_color_hue(2)[1]) +
    geom_line(aes(x=np_iteration, y=np_test_loss),
              colour=gg_color_hue(2)[2]) +
    ggtitle('Nonparametric (Sigmoid) Testing Loss, Parametric Initialization') +
    xlab('Iteration') +
    ylab('Testing Loss')
ggsave(file='np_sigmoid_50_init50.png', width=10, height=6)


# 50k parametric, 50k sin/cos nonparametric
np_sincos_50 <- read.csv('lstm_np_03_loss.csv')
plot_df <- data.frame(init_iteration=parametric_init_df$iteration,
                      init_test_loss=parametric_init_df$testing_loss,
                      np_iteration=np_sincos_50$iteration,
                      np_test_loss=np_sincos_50$testing_loss)
plot_df$init_iteration <- plot_df$init_iteration * 100
plot_df$np_iteration <- plot_df$np_iteration * 100

ggplot(plot_df) +
    geom_line(aes(x=init_iteration, y=init_test_loss),
              colour=gg_color_hue(2)[1]) +
    geom_line(aes(x=np_iteration, y=np_test_loss),
              colour=gg_color_hue(2)[2]) +
    ggtitle('Nonparametric (Sin/Cos) Testing Loss, Parametric Initialization') +
    xlab('Iteration') +
    ylab('Testing Loss')
ggsave(file='np_sincos_50_init50.png', width=10, height=6)

# same as above, but plot training as well
# plot_df <- data.frame(init_iteration=parametric_init_df$iteration,
#                       init_test_loss=parametric_init_df$testing_loss,
#                       init_train_loss=parametric_init_df$training_loss,
#                       np_iteration=np_sincos_50$iteration,
#                       np_test_loss=np_sincos_50$testing_loss,
#                       np_train_loss=np_sincos_50$training_loss)
# plot_df$init_iteration <- plot_df$init_iteration * 100
# plot_df$np_iteration <- plot_df$np_iteration * 100

# ggplot(plot_df) +
#     geom_line(aes(x=init_iteration, y=init_test_loss),
#               colour=gg_color_hue(2)[1]) +
#     geom_line(aes(x=init_iteration, y=init_train_loss),
#               colour='#c0392b', linetype=3) +
#     geom_line(aes(x=np_iteration, y=np_test_loss),
#               colour=gg_color_hue(2)[2]) +
#     geom_line(aes(x=np_iteration, y=np_train_loss),
#               colour='#2980b9', linetype=3) +
#     ggtitle('Test and Train Loss for Nonparametric (Sin/Cos) w/Parametric Initialization') +
#     xlab('Iteration') +
#     ylab('Loss')
# ggsave(file='np_sincos_50_init50_train_test.png', width=10, height=6)

# Plot sin/cos and sigmoid together
plot_df <- data.frame(
  init_iteration=parametric_init_df$iteration,
  init_test_loss=parametric_init_df$testing_loss)
plot_df$init_iteration <- plot_df$init_iteration * 100

# Melt 2x NP
np_df <- data.frame(
  np_iteration=np_sigmoid_50$iteration,
  np_sigmoid_test_loss=np_sigmoid_50$testing_loss,
  np_sincos_test_loss=np_sincos_50$testing_loss)
plot_df2 <- melt(np_df, id='np_iteration')
plot_df2$np_iteration <- plot_df2$np_iteration * 100

final_plot_df <- cbind(plot_df, plot_df2)
# Only keep every 500 iterations
final_plot_df <- final_plot_df %>% filter(init_iteration %% 500 == 0)

# Change colors so that red

ggplot(final_plot_df) +
    geom_line(aes(x=init_iteration,
                  y=init_test_loss),
              colour='#34495e') +
    geom_line(aes(x=np_iteration,
                  y=value,
                  colour=variable)) +
    ggtitle('Nonparametric Testing Loss, Parametric Initialization') +
    xlab('Iteration') +
    ylab('Testing Loss')
ggsave(file='np_together_50_init50.png', width=10, height=6)