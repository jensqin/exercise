library(MASS)
library(GRridge)
library(tidyverse)
library(modelr)

dfcol <- colnames(Boston)
X <- dplyr::select(Boston, -medv)
y <- Boston$medv

df <- crossv_loo(Boston)
# genset <- sapply(1:100,function(x) paste("Gene",x))
# signature <- sapply(seq(1,100,by=2),function(x) paste("Gene",x))
groups <- factor(rep(c("a", "b"), c(5, 8)))
# parts = list(g1 = CreatePartition(groups[1:5], groups),
#              g2 = CreatePartition(groups[6:13], groups))
parts = list(group=CreatePartition(groups))
model1 <- grridge(t(X), y, partitions = parts)

X_test = scale(X[sample(1:nrow(X), 10, replace = F),])
yhat = predict.grridge(model1, datanew=data.frame(t(X_test)))
