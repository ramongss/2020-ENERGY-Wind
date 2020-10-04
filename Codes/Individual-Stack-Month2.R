# Starting Code ----
## Clear memory
rm(list=ls(all=TRUE))
Sys.setenv("LANGUAGE"="En")
Sys.setlocale("LC_ALL", "English")

## Set working directory
setwd("~/Github/2020-ENERGY-Wind/")

BaseDir       <- getwd()
ResultsDir    <- paste(BaseDir, "Results-Month2", sep="/")
DataDir       <- paste(BaseDir, "Data", sep="/")

## Load libraries
library(ggplot2)
library(dplyr)
library(Metrics)
library(mlbench)
library(caret)
library(caretEnsemble)
library(e1071)
library(readxl)
library(forecast)
library(quantregForest)
library(hht)
library(foreach)
library(iterators)
library(doParallel)
library(faraway)

## Cores Cluster
# ncl <- detectCores();ncl
# cl  <- makeCluster(ncl-1);registerDoParallel(cl)
# stopImplicitCluster() # Stop

# Data treatment ----
## Load data
setwd(DataDir)
raw_data <- read_excel('dataset1.xlsx')

dataset <- raw_data[months(raw_data$PCTimeStamp) %in% month.name[9],] %>% as.data.frame()

dataset <- dataset %>% tidyr::drop_na()

dataset$PCTimeStamp <- c(1:dim(dataset)[1])

## dataframes

lag <- 4

dataset.lag <- data.frame(dataset[,2][(lag+1):(dim(dataset)[1]-lag+4)],
                          dataset[,2][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,2][(lag-1):(dim(dataset)[1]-lag+2)],
                          dataset[,2][(lag-2):(dim(dataset)[1]-lag+1)],
                          dataset[,2][(lag-3):(dim(dataset)[1]-lag+0)],
                          dataset[,3][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,4][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,5][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,6][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,7][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,8][(lag-0):(dim(dataset)[1]-lag+3)],
                          dataset[,9][(lag-0):(dim(dataset)[1]-lag+3)])

colnames(dataset.lag) <- c('y','lag1','lag2','lag3','lag4',
                           names(dataset)[3],names(dataset)[4],
                           names(dataset)[5],names(dataset)[6],
                           names(dataset)[7],names(dataset)[8],
                           names(dataset)[9])

n <- dim(dataset.lag)[1]
cut <- 3305

train <- dataset.lag[1:cut,]
test <- tail(dataset.lag,n-cut)

x_train <- train[,-1]
y_train <- train[,1]

x_test <- test[,-1]
y_test <- test[,1]

setwd(ResultsDir)
save.image("lag-data.RData")

# Training and Predictions ----
# setwd(ResultsDir)
# load("lag-data.RData")

set.seed(1234)

control <- trainControl(method = "cv",
                        # initialWindow = 0.8*dim(train)[1],
                        # horizon = 0.1*dim(train)[1],
                        # fixedWindow = FALSE,
                        # allowParallel = TRUE,
                        # savePredictions = 'final',
                        # verboseIter = FALSE)
                        number = 5)

## Set a list of base models
model.list <- c('knn',
                'svmRadial',
                'pls',
                'rf')

{
  model <- list()
  pred.train <- list()
  pred.test  <- list()
  pred <- matrix(ncol = length(model.list), nrow = n)
  
  Metrics.train <- matrix(nrow = length(model.list),ncol = 4)
  Metrics       <- matrix(nrow = length(model.list),ncol = 4)
  colnames(Metrics) <- c("i","MAE","MAPE","RMSE")
  rownames(Metrics) <- model.list
}

for (i in 1:length(model.list)) {
  model[[i]] <- train(y~., data = train,
                      preProcess = c("BoxCox"),
                      method = model.list[i],
                      trControl = control,
                      tuneLength = 5,
                      trace = FALSE)
  
  pred.train[[i]] <- predict(model[[i]],train)
  pred.test[[i]]  <- predict(model[[i]],test)
  pred[,i]   <- c(pred.train[[i]],pred.test[[i]])
  
  MAE  <- MAE(pred.test[[i]], y_test)
  MAPE <- mape(pred.test[[i]], y_test)
  RMSE <- RMSE(pred.test[[i]], y_test)
  
  MAE.train  <- MAE(pred.train[[i]], y_train)
  MAPE.train <- mape(pred.train[[i]], y_train)
  RMSE.train <- RMSE(pred.train[[i]], y_train)
  
  Metrics.train[i,] <- c(i,
                         MAE.train,
                         MAPE.train,
                         RMSE.train)
  Metrics[i,] <- c(i,
                   MAE,
                   MAPE,
                   RMSE)
  
  cat("\nModel:", model.list[i], "MAE:", MAE,"MAPE:", MAPE, 
      "RMSE:", RMSE, as.character(Sys.time()),"\n")
  save.image("Individual-results.RData")
}

Metrics

xtable::xtable(Metrics, digits = 4)

# Stacking Training and Predictions---------------------------------
# setwd(ResultsDir)
# load("Individual-results.RData")
set.seed(1234)
stack.database <- data.frame(dataset.lag$y,pred)
colnames(stack.database) <- c("y",model.list)

stack.database.train <- stack.database[1:cut,]
stack.database.test  <- tail(stack.database,n-cut)

meta.list <- c('cubist')

preprocess.list <- c("corr","pca","BoxCox")

stack <- list()
stack.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                     nrow = n)
colnames(stack.pred) <- c("CORR-CUBIST","PCA-CUBIST","BC-CUBIST")

stack.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                              ncol = 4)
stack.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                              ncol = 4)
colnames(stack.Metrics) <- c("k","MAE","MAPE","RMSE")
rownames(stack.Metrics) <- c("CORR-CUBIST","PCA-CUBIST","BC-CUBIST")

k <- 1

for (i in 1:length(meta.list)) {
  for (j in 1:length(preprocess.list)) {
    stack[[k]] <- train(y~.,data = stack.database.train,
                        method = meta.list[i],
                        trControl = control,
                        preProcess = preprocess.list[j],
                        tuneLength = 5,
                        trace = FALSE)
    
    
    stack.pred.train <- predict(stack[[k]],stack.database.train)
    stack.pred.test  <- predict(stack[[k]],stack.database.test)
    stack.pred[,k]   <- c(stack.pred.train,stack.pred.test)
    
    stack.MAE  <- MAE(stack.pred.test, test$y)
    stack.MAPE <- mape(stack.pred.test, test$y)
    stack.RMSE <- RMSE(stack.pred.test, test$y)
    
    stack.MAE.train  <- MAE(stack.pred.train, train$y)
    stack.MAPE.train <- mape(stack.pred.train, train$y)
    stack.RMSE.train <- RMSE(stack.pred.train, train$y)
    
    stack.Metrics.train[k,] <- c(k,
                                 stack.MAE.train,
                                 stack.MAPE.train,
                                 stack.RMSE.train)
    stack.Metrics[k,] <- c(k,
                           stack.MAE,
                           stack.MAPE,
                           stack.RMSE)
    
    k <- k + 1
    
    cat("Model:", meta.list[i], "pp:", preprocess.list[j], 
        "MAE:", stack.MAE, "MAPE:", stack.MAPE,"RMSE:", stack.RMSE, "\n\n")
  }
  save.image("stack-results.RData")
}

stack.Metrics

xtable::xtable(stack.Metrics, digits = 4)

# Steps-ahead Predictions ----
setwd(ResultsDir)
load("Individual-results.RData")

## 2-step-ahead
{
  {
    h <- 2
    obsTR <- matrix(seq(1,dim(x_train)[1]+1,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(x_test)[1]+1,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(model.list), nrow = dim(train)[1])
    PTEmo     <- matrix(ncol = length(model.list), nrow = dim(test)[1])
    Pred2step <- matrix(ncol = length(model.list), nrow = n)
    colnames(PTRmo) <- model.list
    colnames(PTEmo) <- model.list
    Metrics2.train <- matrix(nrow = length(model.list), ncol = 4)
    Metrics2       <- matrix(nrow = length(model.list), ncol = 4)
    colnames(Metrics2.train) <- c("m","MAE","MAPE","RMSE") 
    colnames(Metrics2)       <- c("m","MAE","MAPE","RMSE")
    rownames(Metrics2)       <- model.list
  }
  
  for (m in 1:length(model.list)) {
    x_trainm <- as.data.frame(x_train)
    x_testm  <- as.data.frame(x_test)
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]){
            PTRmo[p,m] <- predict(model[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(model[[m]],x_trainm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE)[1]) {
        p <- obsTE[v,N]
        if (p <= dim(x_testm)[1]) {
          if(p==obsTE[v,1]){
            PTEmo[p,m] <- predict(model[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(model[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    Pred2step[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred2.MAE.train  <- MAE(PTRmo[,m], train[,1])
    pred2.MAPE.train <- mape(PTRmo[,m], train[,1])
    pred2.RMSE.train <- RMSE(PTRmo[,m], train[,1])
    
    pred2.MAE  <- MAE(PTEmo[,m], test[,1])
    pred2.MAPE <- mape(PTEmo[,m], test[,1])
    pred2.RMSE <- RMSE(PTEmo[,m], test[,1])
    
    
    Metrics2.train[m,] <- c(m,pred2.MAE.train,pred2.MAPE.train,pred2.RMSE.train)
    Metrics2[m,] <- c(m,pred2.MAE,pred2.MAPE,pred2.RMSE)
    
    cat("Model: ", model.list[m]," ",m/length(model.list)*100, "%\n", sep = "")
  }
  
  xtable::xtable(Metrics2, digits = 4)
  
  save.image("2step-Individual-results.RData")
}

## 3-step-ahead
{
{
  h <- 3
  obsTR <- matrix(seq(1,dim(x_train)[1]+1,1),ncol = h, byrow = TRUE)
  obsTE <- matrix(seq(1,dim(x_test)[1]+2,1),ncol = h, byrow = TRUE)
  PTRmo <- matrix(ncol = length(model.list), nrow = dim(train)[1])
  PTEmo     <- matrix(ncol = length(model.list), nrow = dim(test)[1])
  Pred3step <- matrix(ncol = length(model.list), nrow = n)
  colnames(PTRmo) <- model.list
  colnames(PTEmo) <- model.list
  Metrics3.train <- matrix(nrow = length(model.list), ncol = 4)
  Metrics3       <- matrix(nrow = length(model.list), ncol = 4)
  colnames(Metrics3.train) <- c("m","MAE","MAPE","RMSE") 
  colnames(Metrics3)       <- c("m","MAE","MAPE","RMSE")
  rownames(Metrics3)       <- model.list
}

for (m in 1:length(model.list)) {
  x_trainm <- as.data.frame(x_train)
  x_testm  <- as.data.frame(x_test)
  
  # Train
  for (N in 1:h) {
    for (v in 1:dim(obsTR)[1]) {
      p <- obsTR[v,N]
      if (p <= dim(x_trainm)[1]) {
        if(p==obsTR[v,1]){
          PTRmo[p,m] <- predict(model[[m]],x_trainm[p,])
        }
        else if(p==obsTR[v,2]) {
          x_trainm[p,1] <- PTRmo[p-1,m]
          PTRmo[p,m]    <- predict(model[[m]],x_trainm[p,])
        }
        else {
          x_trainm[p,1] <- PTRmo[p-1,m]
          x_trainm[p,2] <- x_trainm[p-1,1]
          
          PTRmo[p,m]   <-predict(model[[m]],x_trainm[p,])
        }
      }
      else {
        break
      }
    }
  }
  
  # Test
  for (N in 1:h) {
    for (v in 1:dim(obsTE)[1]) {
      p <- obsTE[v,N]
      if (p <= dim(x_testm)[1]) {
        if(p==obsTE[v,1]){
          PTEmo[p,m] <- predict(model[[m]],x_testm[p,])
        }
        else if(p==obsTE[v,2]) {
          x_testm[p,1] <- PTEmo[p-1,m]
          PTEmo[p,m]    <- predict(model[[m]],x_testm[p,])
        }
        else {
          x_testm[p,1] <- PTEmo[p-1,m]
          x_testm[p,2] <- x_testm[p-1,1]
          
          PTEmo[p,m]   <-predict(model[[m]],x_testm[p,])
        }
      }
      else {
        break
      }
    }
  }
  
  
  Pred3step[,m] <- c(PTRmo[,m],PTEmo[,m])
  
  ### Metrics
  
  pred3.MAE.train  <- MAE(PTRmo[,m], train[,1])
  pred3.MAPE.train <- mape(PTRmo[,m], train[,1])
  pred3.RMSE.train <- RMSE(PTRmo[,m], train[,1])
  
  pred3.MAE  <- MAE(PTEmo[,m], test[,1])
  pred3.MAPE <- mape(PTEmo[,m], test[,1])
  pred3.RMSE <- RMSE(PTEmo[,m], test[,1])
  
  
  Metrics3.train[m,] <- c(m,pred3.MAE.train,pred3.MAPE.train,pred3.RMSE.train)
  Metrics3[m,] <- c(m,pred3.MAE,pred3.MAPE,pred3.RMSE)
  
  cat("Model: ", model.list[m]," ",m/length(model.list)*100, "%\n", sep = "")
}

xtable::xtable(Metrics3, digits = 4)

save.image("3step-Individual-results.RData")
}

# Stacking steps-ahead Predictions ----

## 2-steps-ahead # CORRETO
{
  # setwd(ResultsDir)
  # load("2step-Individual-results.RData")
  
  set.seed(1234)
  stack2.database <- data.frame(dataset.lag[,1],Pred2step)
  colnames(stack2.database) <- c("y",model.list)
  
  stack2.database.train <- stack2.database[1:cut,]
  stack2.database.test  <- tail(stack2.database,n-cut)
  
  meta.list <- c("cubist")
  
  preprocess.list <- c("corr","pca","BoxCox")
  
  stack <- list()
  
  for (p in 1:length(preprocess.list)) {
    stack[[p]] <- train(y~.,data = stack2.database.train,
                        method = meta.list,
                        trControl = control,
                        preProcess = preprocess.list[p],
                        trace = FALSE)
    cat("PP: ", preprocess.list[p], "\t", p/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  {
    h <- 2
    obsTR <- matrix(seq(1,dim(train)[1]+1,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(test)[1]+1,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = dim(train)[1])
    PTEmo     <- matrix(ncol = length(preprocess.list), nrow = dim(test)[1])
    colnames(PTRmo) <- preprocess.list
    colnames(PTEmo) <- preprocess.list
    stack2.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                          nrow = n)
    stack2.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    stack2.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    colnames(stack2.Metrics) <- c("m","MAE","MAPE","RMSE")
    rownames(stack2.Metrics) <- preprocess.list
  }
  
  for (m in 1:length(preprocess.list)) {
    x_trainm <- as.data.frame(stack2.database.train[,-1])
    x_testm  <- as.data.frame(stack2.database.test[,-1])
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]){
            PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE)[1]) {
        p <- obsTE[v,N]
        if (p <= dim(x_testm)[1]) {
          if(p==obsTE[v,1]){
            PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    stack2.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred2.MAE.train  <- MAE(PTRmo[,m], train[,1])
    pred2.MAPE.train <- mape(PTRmo[,m], train[,1])
    pred2.RMSE.train <- RMSE(PTRmo[,m], train[,1])
    
    pred2.MAE  <- MAE(PTEmo[,m], test[,1])
    pred2.MAPE <- mape(PTEmo[,m], test[,1])
    pred2.RMSE <- RMSE(PTEmo[,m], test[,1])
    
    
    stack2.Metrics.train[m,] <- c(m,pred2.MAE.train,pred2.MAPE.train,pred2.RMSE.train)
    stack2.Metrics[m,] <- c(m,pred2.MAE,pred2.MAPE,pred2.RMSE)
    
    cat("stack: ", preprocess.list[m]," ",m/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  save.image("2steps-stack-results.RData")
  
  xtable::xtable(stack2.Metrics, digits = 4)
  
}

## 3-steps-ahead # CORRETO
{
  # setwd(ResultsDir)
  # load("3step-Individual-results.RData")
  
  set.seed(1234)
  stack3.database <- data.frame(dataset.lag[,1],Pred3step)
  colnames(stack3.database) <- c("y",model.list)
  
  stack3.database.train <- stack3.database[1:cut,]
  stack3.database.test  <- tail(stack3.database,n-cut)
  
  meta.list <- c("cubist")
  
  preprocess.list <- c("corr","pca","BoxCox")
  
  stack <- list()
  
  for (p in 1:length(preprocess.list)) {
    stack[[p]] <- train(y~.,data = stack3.database.train,
                        method = meta.list,
                        trControl = control,
                        preProcess = preprocess.list[p],
                        trace = FALSE)
    cat("PP: ", preprocess.list[p], "\t", p/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  {
    h <- 3
    obsTR <- matrix(seq(1,dim(train)[1]+1,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,dim(test)[1]+2,1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = dim(train)[1])
    PTEmo     <- matrix(ncol = length(preprocess.list), nrow = dim(test)[1])
    colnames(PTRmo) <- preprocess.list
    colnames(PTEmo) <- preprocess.list
    stack3.pred <- matrix(ncol = length(meta.list)*length(preprocess.list), 
                          nrow = n)
    stack3.Metrics.train <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    stack3.Metrics       <- matrix(nrow = length(meta.list)*length(preprocess.list),
                                   ncol = 4)
    colnames(stack3.Metrics) <- c("m","MAE","MAPE","RMSE")
    rownames(stack3.Metrics) <- preprocess.list
  }
  
  for (m in 1:length(preprocess.list)) {
    x_trainm <- as.data.frame(stack3.database.train[,-1])
    x_testm  <- as.data.frame(stack3.database.test[,-1])
    
    # Train
    for (N in 1:h) {
      for (v in 1:dim(obsTR)[1]) {
        p <- obsTR[v,N]
        if (p <= dim(x_trainm)[1]) {
          if(p==obsTR[v,1]){
            PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
          }
          else if(p==obsTR[v,2]) {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            x_trainm[p,2] <- x_trainm[p-1,1]
            
            PTRmo[p,m]   <-predict(stack[[m]],x_trainm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    # Test
    for (N in 1:h) {
      for (v in 1:dim(obsTE)[1]) {
        p <- obsTE[v,N]
        if (p <= dim(x_testm)[1]) {
          if(p==obsTE[v,1]){
            PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
          }
          else if(p==obsTE[v,2]) {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            x_testm[p,2] <- x_testm[p-1,1]
            
            PTEmo[p,m]   <-predict(stack[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    stack3.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred3.MAE.train  <- MAE(PTRmo[,m], train[,1])
    pred3.MAPE.train <- mape(PTRmo[,m], train[,1])
    pred3.RMSE.train <- RMSE(PTRmo[,m], train[,1])
    
    pred3.MAE  <- MAE(PTEmo[,m], test[,1])
    pred3.MAPE <- mape(PTEmo[,m], test[,1])
    pred3.RMSE <- RMSE(PTEmo[,m], test[,1])
    
    
    stack3.Metrics.train[m,] <- c(m,pred3.MAE.train,pred3.MAPE.train,pred3.RMSE.train)
    stack3.Metrics[m,] <- c(m,pred3.MAE,pred3.MAPE,pred3.RMSE)
    
    cat("stack: ", preprocess.list[m]," ",m/length(preprocess.list)*100, "%\n", sep = "")
  }
  
  save.image("3steps-stack-results.RData")
  
  xtable::xtable(stack3.Metrics, digits = 4)

}
