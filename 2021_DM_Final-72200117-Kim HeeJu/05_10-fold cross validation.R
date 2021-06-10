library(caret) # createFolds
library(e1071) # svm


###10-fold cross validation evaluate function

evaluator <- function(subset) {
  set.seed(100) 
  K<- 10
  folds <- createFolds(1:nrow(ds), K)
  acc <- c()
  for (i in 1:K) {
    ts.idx <- folds[[i]]
    ds.tr <- ds[-ts.idx,subset]
    ds.ts <- ds[ts.idx,subset]
    cl.tr <- cl[-ts.idx]
    cl.ts <- cl[ts.idx]
    model <- svm(ds.tr,cl.tr)
    pred <- predict(model, ds.ts)
    acc[i] <- mean(pred==cl.ts)
  }
  cat(mean(acc),subset, "\n")
  return(mean(acc))
}


Kfold.eval <- function(ds, cl, subset) {
  set.seed(100)
  k=5 # folds
  folds <- createFolds(1:nrow(ds), k)
  acc <- c()
  for (i in 1:k) {
    ts.idx <- folds[[i]]
    ds.tr <- ds[-ts.idx,subset]
    ds.ts <- ds[ts.idx,subset]
    cl.tr <- cl[-ts.idx]
    cl.ts <- cl[ts.idx]
    model <- svm(ds.tr,cl.tr)
    pred <- predict(model, ds.ts)
    acc[i] <- mean(pred==cl.ts)
  }
  cat(mean(acc),subset, "\n")
  return(mean(acc))
}



myCV <- function (ds, cl, K=10) {
  set.seed(100)
  folds <- createFolds(1:nrow(ds), k = K)
  acc <- c()
  for (i in 1:K) {
    ts.idx <- folds[[i]]
    ds.tr <- ds[-ts.idx,]
    ds.ts <- ds[ts.idx,]
    cl.tr <- cl[-ts.idx]
    cl.ts <- cl[ts.idx]
    model <- svm(ds.tr,cl.tr)
    pred <- predict(model, ds.ts)
    acc[i] <- mean(pred==cl.ts)
  }
  return(mean(acc))
}

