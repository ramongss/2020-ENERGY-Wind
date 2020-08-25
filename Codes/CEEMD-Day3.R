# Starting Code ----
## Clear memory
rm(list=ls(all=TRUE))
Sys.setenv("LANGUAGE"="En")
Sys.setlocale("LC_ALL", "English")

## Set working directory
setwd("~/Github/2020-ENERGY-Wind/")

BaseDir       <- getwd()
ResultsDir    <- paste(BaseDir, "Results-Day3", sep="/")
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

# Data Treatment ----
## Load data
setwd(DataDir)
raw_data <- read_excel('dataset.xlsx')

dataset <- data.frame(raw_data[865:1008,])

dataset$PCTimeStamp <- c(1:dim(dataset)[1])

## Decompose data
ceemd <- CEEMD(sig = dataset$Power, tt = dataset$PCTimeStamp,
               noise.amp = (sd(dataset$Power)*.25),
               trials = 100, verbose = FALSE, max.imf = 5)
# PlotIMFs(ceemd)

## PACF and auto.arima 
PACF <- list()
ARIMA <- list()

for (i in seq(ceemd$nimf)) {
  PACF[[i]] <- pacf(ceemd$imf[,i], main = paste("IMF ",i))
  ARIMA[[i]] <- auto.arima(ceemd$imf[,i])
  print(ARIMA[[i]])
}
PACF[[ceemd$nimf+1]] <- pacf(ceemd$residue, main = paste("Residue"))
ARIMA[[ceemd$nimf+1]] <- auto.arima(ceemd$residue)
print(ARIMA[[ceemd$nimf+1]])


IMF <- matrix(ncol = ceemd$nimf+1, nrow = dim(dataset)[1])
colnames(IMF) <- c("IMF1","IMF2","IMF3","IMF4","IMF5","Residue")
for (i in seq(ceemd$nimf)) {
  IMF[,i] <- ceemd$imf[,i]
}
IMF[,ceemd$nimf+1] <- ceemd$residue


## Dataframes

lag <- 4

IMF_df <- list()

for (i in seq(dim(IMF)[2])) {
  IMF_df[[i]] <- data.frame(IMF[,i][(lag+1):(dim(IMF)[1]-lag+4)],
                            IMF[,i][(lag-0):(dim(IMF)[1]-lag+3)],
                            IMF[,i][(lag-1):(dim(IMF)[1]-lag+2)],
                            IMF[,i][(lag-2):(dim(IMF)[1]-lag+1)],
                            IMF[,i][(lag-3):(dim(IMF)[1]-lag+0)],
                            dataset[,3][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,4][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,5][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,6][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,7][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,8][(lag-0):(dim(IMF)[1]-lag+3)],
                            dataset[,9][(lag-0):(dim(IMF)[1]-lag+3)])
  colnames(IMF_df[[i]]) <- c('y','lag1','lag2','lag3','lag4',
                             names(dataset)[3],names(dataset)[4],
                             names(dataset)[5],names(dataset)[6],
                             names(dataset)[7],names(dataset)[8],
                             names(dataset)[9])
}

## Training and Test sets
IMF.train  <- list()
IMF.test   <- list()
IMF.xtrain <- list()
IMF.ytrain <- list()
IMF.xtest  <- list()
IMF.ytest  <- list()

for (i in seq(IMF_df)) {
  n <- dim(IMF_df[[i]])[1]
  cut <- 0.7*n
  
  IMF.train[[i]] <- IMF_df[[i]][1:cut,]
  IMF.test[[i]]  <- tail(IMF_df[[i]],n-cut)
  
  IMF.xtrain[[i]] <- IMF.train[[i]][,-1]
  IMF.ytrain[[i]] <- IMF.train[[i]][,1]
  
  IMF.xtest[[i]] <- IMF.test[[i]][,-1]
  IMF.ytest[[i]] <- IMF.test[[i]][,1]
}

Obs   <- dataset$Power[(lag+1):(dim(IMF)[1]-lag+4)]
Obs.train <- Obs[1:cut]
Obs.test  <- tail(Obs,n-cut)

setwd(ResultsDir)
save.image("CEEMD-data.RData")

# CEEMD Training and Prediction ----
setwd(ResultsDir)
load('CEEMD-data.RData')

## Set random seed
set.seed(1234)

## Set traincontrol
control <- trainControl(method = "timeslice",
                        initialWindow = 0.8*dim(IMF.train[[1]])[1],
                        horizon = 0.1*dim(IMF.train[[1]])[1],
                        fixedWindow = FALSE,
                        allowParallel = TRUE,
                        savePredictions = 'final',
                        verboseIter = FALSE)

## Set a list of base models
model.list <- c('knn',
                'svmRadial',
                'pls',
                'rf')

## Define objects 
{
  IMF.model      <- list()
  pred.IMF       <- list()
  k <- 1
}

## Training and Predicting each IMF with each model
for (i in seq(model.list)) {
  for (j in seq(IMF_df)) {
    IMF.model[[k]] <- train(y~., data = IMF.train[[j]],
                            method = model.list[i],
                            trControl = control,
                            preProcess = c("BoxCox"),
                            tuneLenght = 5,
                            trace = FALSE)
    
    ### Prediction
    
    pred.IMF.train   <- predict(IMF.model[[k]],IMF.train[[j]])
    pred.IMF.test    <- predict(IMF.model[[k]],IMF.test[[j]])
    pred.IMF[[k]]    <- data.frame(c(pred.IMF.train,pred.IMF.test))
    
    cat("\nModel: ", model.list[i], "\tIMF", j, "\t",
        format(strptime(Sys.time(),"%Y-%m-%d %H:%M:%S"),'%H:%M:%S'), 
        sep = '')
    
    k <- k + 1
  }
}

## Create a matrix combination
combs <- matrix(nrow = length(model.list), ncol = length(IMF_df))
colnames(combs) <- colnames(IMF)

for (i in seq(model.list)) {
  for (j in seq(IMF_df)) {
    combs[i,j] <- i
  }
}

## Define objects to Evaluate Performance
{
  CEEMD.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
  CEEMD.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
  colnames(CEEMD.Metrics)       <- c("i","MAE","MAPE","RMSE")
  colnames(CEEMD.Metrics.train) <- c("i","MAE","MAPE","RMSE")
  rownames(CEEMD.Metrics)       <- model.list
  rownames(CEEMD.Metrics.train) <- model.list
  
  CEEMD.Prediction <- matrix(0,nrow = n, ncol = dim(combs)[1])
  colnames(CEEMD.Prediction) <- model.list
  
  k <- 1
}

## Recompose dataset by summing IMF
for (i in seq(model.list)) {
  for (j in seq(IMF_df)) {
    CEEMD.Prediction[,i] <- as.vector(CEEMD.Prediction[,i]) + 
      as.vector(pred.IMF[[k]][,1])
    k <- k + 1
  }
}

## Evaluate Performance
for (i in seq(model.list)) {
  
  ## Avoid negative predictions
  for (j in seq(n)) {
    if (CEEMD.Prediction[j,i] < 0) {
      CEEMD.Prediction[j,i] <- 0
    }
  }
  
  CEEMD.Prediction.train <- CEEMD.Prediction[,i][1:cut]
  CEEMD.Prediction.test  <- tail(CEEMD.Prediction[,i],n-cut)
  
  # #Metrics
  CEEMD.MAE   <- MAE(CEEMD.Prediction.test, Obs.test)
  CEEMD.MAPE <- mape(CEEMD.Prediction.test, Obs.test)
  CEEMD.RMSE <- RMSE(CEEMD.Prediction.test, Obs.test)
  
  CEEMD.MAE.train  <- MAE(CEEMD.Prediction.train, Obs.train)
  CEEMD.MAPE.train <- mape(CEEMD.Prediction.train, Obs.train)
  CEEMD.RMSE.train <- RMSE(CEEMD.Prediction.train, Obs.train)
  
  CEEMD.Metrics.train[i,] <- c(i,
                               CEEMD.MAE.train,
                               CEEMD.MAPE.train,
                               CEEMD.RMSE.train)
  CEEMD.Metrics[i,] <- c(i,
                         CEEMD.MAE,
                         CEEMD.MAPE,
                         CEEMD.RMSE)
}

save.image("CEEMD-results.RData")

CEEMD.Metrics
# xtable::xtable(CEEMD.Metrics, digits = 4)

# CEEMD-Stack Training and Predictions ----
## Set random seed
set.seed(1234)

## Create dataframe for stack
stack.database <- data.frame(Obs,CEEMD.Prediction)
colnames(stack.database) <- c("y",model.list)

## Split new dataframe into train-test
stack.database.train <- stack.database[1:cut,]
stack.database.test  <- tail(stack.database,n-cut)

## Set a list of meta models
meta.list <- c('svmLinear2','brnn','cubist')

## Set a list of preprocessing techniques
pp.list <- c("corr","pca","BoxCox")

## Define objects
stack <- list()
stack.pred <- matrix(ncol = length(meta.list)*length(pp.list), 
                     nrow = n)
colnames(stack.pred) <- c("CORR-SVRL","PCA-SVRL","BC-SVRL",
                          "CORR-BRNN","PCA-BRNN","BC-BRNN",
                          "CORR-CUBIST","PCA-CUBIST","BC-CUBIST")
stack.Metrics.train <- matrix(nrow = length(meta.list)*length(pp.list),
                              ncol = 4)
stack.Metrics       <- matrix(nrow = length(meta.list)*length(pp.list),
                              ncol = 4)
colnames(stack.Metrics) <- c("k","MAE","MAPE","RMSE")
rownames(stack.Metrics) <- c("CORR-SVRL","PCA-SVRL","BC-SVRL",
                             "CORR-BRNN","PCA-BRNN","BC-BRNN",
                             "CORR-CUBIST","PCA-CUBIST","BC-CUBIST")
k <- 1

for (i in 1:length(meta.list)) {
  for (j in 1:length(pp.list)) {
    stack[[k]] <- train(y~.,data = stack.database.train,
                        method = meta.list[i],
                        trControl = control,
                        preProcess = pp.list[j],
                        tuneLength = 5,
                        trace = FALSE)
    
    
    stack.pred.train <- predict(stack[[k]],stack.database.train)
    stack.pred.test  <- predict(stack[[k]],stack.database.test)
    stack.pred[,k]   <- c(stack.pred.train,stack.pred.test)
    
    
    stack.MAE   <- MAE(stack.pred.test, Obs.test)
    stack.MAPE  <- mape(stack.pred.test, Obs.test)
    stack.RMSE  <- RMSE(stack.pred.test, Obs.test)
    
    stack.MAE.train  <- MAE(stack.pred.train, Obs.train)
    stack.MAPE.train <- mape(stack.pred.train, Obs.train)
    stack.RMSE.train <- RMSE(stack.pred.train, Obs.train)
    
    stack.Metrics.train[k,] <- c(k,
                                 stack.MAE.train,
                                 stack.MAPE.train,
                                 stack.RMSE.train)
    stack.Metrics[k,] <- c(k,
                           stack.MAE,
                           stack.MAPE,
                           stack.RMSE)
    
    k <- k + 1
    
    cat("Model:", meta.list[i], "pp:", pp.list[j],
        "MAE:", stack.MAE, "MAPE:", stack.MAPE, "RMSE:", stack.RMSE, 
        format(strptime(Sys.time(),"%Y-%m-%d %H:%M:%S"),'%H:%M:%S'),"\n\n")
  }
  save.image("CEEMD-STACK-results.RData")
}

stack.Metrics

xtable::xtable(stack.Metrics, digits = 4)

# CEEMD Steps-ahead Prediction ----
setwd(ResultsDir)
load("CEEMD-results.RData")

## 2 steps
{
  {
    IMF1.model <- list(IMF.model[[1]],
                       IMF.model[[7]],
                       IMF.model[[13]],
                       IMF.model[[19]])
    
    IMF2.model <- list(IMF.model[[2]],
                       IMF.model[[8]],
                       IMF.model[[14]],
                       IMF.model[[20]])
    
    IMF3.model <- list(IMF.model[[3]],
                       IMF.model[[9]],
                       IMF.model[[15]],
                       IMF.model[[21]])
    
    IMF4.model <- list(IMF.model[[4]],
                       IMF.model[[10]],
                       IMF.model[[16]],
                       IMF.model[[22]])
    
    IMF5.model <- list(IMF.model[[5]],
                       IMF.model[[11]],
                       IMF.model[[17]],
                       IMF.model[[23]])
    
    IMF6.model <- list(IMF.model[[6]],
                       IMF.model[[12]],
                       IMF.model[[16]],
                       IMF.model[[24]])
  }
  
  ### Recursive prediction
  
  {
    h <- 2
    obsTR <- list()
    obsTE <- list()
    PTRmo <- list()
    PTEmo <- list()
    model <- list()
    Comp.train <- list()
    Comp.test <- list()
    k <- 1
  }
  
  for (m in 1:length(model.list)) {
    
    IMF1.trainm <- as.data.frame(IMF.xtrain[[1]])
    IMF2.trainm <- as.data.frame(IMF.xtrain[[2]])
    IMF3.trainm <- as.data.frame(IMF.xtrain[[3]])
    IMF4.trainm <- as.data.frame(IMF.xtrain[[4]])
    IMF5.trainm <- as.data.frame(IMF.xtrain[[5]])
    IMF6.trainm <- as.data.frame(IMF.xtrain[[6]])
    
    Comp.train[[m]] <- list(IMF1.trainm,
                            IMF2.trainm,
                            IMF3.trainm,
                            IMF4.trainm,
                            IMF5.trainm,
                            IMF6.trainm)
    
    IMF1.testm <- as.data.frame(IMF.xtest[[1]])
    IMF2.testm <- as.data.frame(IMF.xtest[[2]])
    IMF3.testm <- as.data.frame(IMF.xtest[[3]])
    IMF4.testm <- as.data.frame(IMF.xtest[[4]])
    IMF5.testm <- as.data.frame(IMF.xtest[[5]])
    IMF6.testm <- as.data.frame(IMF.xtest[[6]])
    
    Comp.test[[m]] <- list(IMF1.testm,
                           IMF2.testm,
                           IMF3.testm,
                           IMF4.testm,
                           IMF5.testm,
                           IMF6.testm)
    
    model[[m]] <- list(IMF1.model[[m]],IMF2.model[[m]],IMF3.model[[m]],
                       IMF4.model[[m]],IMF5.model[[m]],IMF6.model[[m]])
    
    PTRmo[[m]] <- matrix(ncol = length(IMF_df), nrow = dim(IMF1.trainm)[1])
    PTEmo[[m]] <- matrix(ncol = length(IMF_df), nrow = dim(IMF1.testm)[1])
    
    for (c in 1:length(Comp.train[[m]])) {
      obsTR[[c]] <- matrix(seq(1,dim(Comp.train[[m]][[c]])[1],1),ncol = h, byrow = TRUE)
      obsTE[[c]] <- matrix(seq(1,dim(Comp.test[[m]][[c]])[1],1),ncol = h, byrow = TRUE)
      
      # Train
      for (N in 1:h) {
        for (v in 1:dim(obsTR[[c]])[1]) {
          p <- obsTR[[c]][v,N]
          if (p <= dim(Comp.train[[m]][[c]])[1]) {
            if(p==obsTR[[c]][v,1]){
              PTRmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
            else {
              Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
              PTRmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      # Test
      for (N in 1:h) {
        for (v in 1:dim(obsTE[[c]])[1]) {
          p <- obsTE[[c]][v,N]
          if (p <= dim(Comp.test[[m]][[c]])[1]) {
            if(p==obsTE[[c]][v,1]){
              PTEmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
            else {
              Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
              PTEmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      cat("Model: ", model.list[m], "\tComp: ", c , "\t", 
          (k/(length(model.list)*length(Comp.train[[m]])))*100,"%\n", sep = "")
      
      k <- k + 1
    }
  }
  
  {
    CEEMD2.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
    CEEMD2.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
    colnames(CEEMD2.Metrics)       <- c("i","MAE","MAPE","RMSE")
    colnames(CEEMD2.Metrics.train) <- c("i","MAE","MAPE","RMSE")
    rownames(CEEMD2.Metrics)       <- model.list
    rownames(CEEMD2.Metrics.train) <- model.list
    
    
    CEEMD2.Pred.train <- matrix(nrow = dim(IMF.xtrain[[1]])[1], ncol = dim(combs)[1])
    CEEMD2.Pred.test  <- matrix(nrow = dim(IMF.xtest[[1]])[1], ncol = dim(combs)[1])
    CEEMD2.Prediction <- matrix(nrow = n, ncol = dim(combs)[1])
  }
  
  for (i in 1:dim(combs)[1]) {
    CEEMD2.Pred.train[,i] <- (PTRmo[[combs[i,1]]][,1] + 
                                PTRmo[[combs[i,2]]][,2] + 
                                PTRmo[[combs[i,3]]][,3] +
                                PTRmo[[combs[i,4]]][,4] + 
                                PTRmo[[combs[i,5]]][,5] +
                                PTRmo[[combs[i,6]]][,6])
    
    CEEMD2.Pred.test[,i] <- (PTEmo[[combs[i,1]]][,1] + 
                               PTEmo[[combs[i,2]]][,2] + 
                               PTEmo[[combs[i,3]]][,3] +
                               PTEmo[[combs[i,4]]][,4] + 
                               PTEmo[[combs[i,5]]][,5] +
                               PTEmo[[combs[i,6]]][,6])
    
    ### Avoiding negative values
    for (j in 1:dim(CEEMD2.Pred.train)[1]) {
      if (CEEMD2.Pred.train[j,i] < 0) {
        CEEMD2.Pred.train[j,i] <- 0
      }
    }
    for (j in 1:dim(CEEMD2.Pred.test)[1]) {
      if (CEEMD2.Pred.test[j,i] < 0) {
        CEEMD2.Pred.test[j,i] <- 0
      }
    }
    
    CEEMD2.Prediction[,i] <- c(CEEMD2.Pred.train[,i],CEEMD2.Pred.test[,i])
    
    # #Metrics
    CEEMD2.MAE  <- MAE(CEEMD2.Pred.test[,i], Obs.test)
    CEEMD2.MAPE <- mape(CEEMD2.Pred.test[,i], Obs.test)
    CEEMD2.RMSE <- RMSE(CEEMD2.Pred.test[,i], Obs.test)
    
    CEEMD2.MAE.train  <- MAE(CEEMD2.Pred.train[,i], Obs.train)
    CEEMD2.MAPE.train <- mape(CEEMD2.Pred.train[,i], Obs.train)
    CEEMD2.RMSE.train <- RMSE(CEEMD2.Pred.train[,i], Obs.train)
    
    CEEMD2.Metrics.train[i,] <- c(i,
                                  CEEMD2.MAE.train,
                                  CEEMD2.MAPE.train,
                                  CEEMD2.RMSE.train)
    CEEMD2.Metrics[i,] <- c(i,
                            CEEMD2.MAE,
                            CEEMD2.MAPE,
                            CEEMD2.RMSE)
  }
  
  xtable::xtable(CEEMD2.Metrics, digits = 4)
  
  save.image("2steps-CEEMD-results.RData")
}

## 3 steps
{
  {
    IMF1.model <- list(IMF.model[[1]],
                       IMF.model[[7]],
                       IMF.model[[13]],
                       IMF.model[[19]])
    
    IMF2.model <- list(IMF.model[[2]],
                       IMF.model[[8]],
                       IMF.model[[14]],
                       IMF.model[[20]])
    
    IMF3.model <- list(IMF.model[[3]],
                       IMF.model[[9]],
                       IMF.model[[15]],
                       IMF.model[[21]])
    
    IMF4.model <- list(IMF.model[[4]],
                       IMF.model[[10]],
                       IMF.model[[16]],
                       IMF.model[[22]])
    
    IMF5.model <- list(IMF.model[[5]],
                       IMF.model[[11]],
                       IMF.model[[17]],
                       IMF.model[[23]])
    
    IMF6.model <- list(IMF.model[[6]],
                       IMF.model[[12]],
                       IMF.model[[16]],
                       IMF.model[[24]])
  }
  
  ### Recursive prediction
  
  {
    h <- 3
    obsTR <- list()
    obsTE <- list()
    PTRmo <- list()
    PTEmo <- list()
    model <- list()
    Comp.train <- list()
    Comp.test <- list()
    k <- 1
  }
  
  for (m in 1:length(model.list)) {
    
    IMF1.trainm <- as.data.frame(IMF.xtrain[[1]])
    IMF2.trainm <- as.data.frame(IMF.xtrain[[2]])
    IMF3.trainm <- as.data.frame(IMF.xtrain[[3]])
    IMF4.trainm <- as.data.frame(IMF.xtrain[[4]])
    IMF5.trainm <- as.data.frame(IMF.xtrain[[5]])
    IMF6.trainm <- as.data.frame(IMF.xtrain[[6]])
    
    Comp.train[[m]] <- list(IMF1.trainm,
                            IMF2.trainm,
                            IMF3.trainm,
                            IMF4.trainm,
                            IMF5.trainm,
                            IMF6.trainm)
    
    IMF1.testm <- as.data.frame(IMF.xtest[[1]])
    IMF2.testm <- as.data.frame(IMF.xtest[[2]])
    IMF3.testm <- as.data.frame(IMF.xtest[[3]])
    IMF4.testm <- as.data.frame(IMF.xtest[[4]])
    IMF5.testm <- as.data.frame(IMF.xtest[[5]])
    IMF6.testm <- as.data.frame(IMF.xtest[[6]])
    
    Comp.test[[m]] <- list(IMF1.testm,
                           IMF2.testm,
                           IMF3.testm,
                           IMF4.testm,
                           IMF5.testm,
                           IMF6.testm)
    
    model[[m]] <- list(IMF1.model[[m]],IMF2.model[[m]],IMF3.model[[m]],
                       IMF4.model[[m]],IMF5.model[[m]],IMF6.model[[m]])
    
    PTRmo[[m]] <- matrix(ncol = length(IMF_df), nrow = dim(IMF1.trainm)[1])
    PTEmo[[m]] <- matrix(ncol = length(IMF_df), nrow = dim(IMF1.testm)[1])
    
    for (c in 1:length(Comp.train[[m]])) {
      obsTR[[c]] <- matrix(seq(1,dim(Comp.train[[m]][[c]])[1]+1,1),ncol = h, byrow = TRUE)
      obsTE[[c]] <- matrix(seq(1,dim(Comp.test[[m]][[c]])[1],1),ncol = h, byrow = TRUE)
      
      # Train
      for (N in 1:h) {
        for (v in 1:dim(obsTR[[c]])[1]) {
          p <- obsTR[[c]][v,N]
          if (p <= dim(Comp.train[[m]][[c]])[1]) {
            if(p==obsTR[[c]][v,1]){
              PTRmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
            else if(p==obsTR[[c]][v,2]) {
              Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
              PTRmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
            else {
              Comp.train[[m]][[c]][p,1] <- PTRmo[[m]][p-1,c]
              Comp.train[[m]][[c]][p,2] <- Comp.train[[m]][[c]][p-1,1]
              
              PTRmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.train[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      # Test
      for (N in 1:h) {
        for (v in 1:dim(obsTE[[c]])[1]) {
          p <- obsTE[[c]][v,N]
          if (p <= dim(Comp.test[[m]][[c]])[1]) {
            if(p==obsTE[[c]][v,1]){
              PTEmo[[m]][p,c] <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
            else if(p==obsTE[[c]][v,2]) {
              Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
              PTEmo[[m]][p,c]    <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
            else {
              Comp.test[[m]][[c]][p,1] <- PTEmo[[m]][p-1,c]
              Comp.test[[m]][[c]][p,2] <- Comp.test[[m]][[c]][p-1,1]
              
              PTEmo[[m]][p,c]   <- predict(model[[m]][[c]],Comp.test[[m]][[c]][p,])
            }
          }
          else {
            break
          }
        }
      }
      
      cat("Model: ", model.list[m], "\tComp: ", c , "\t", 
          (k/(length(model.list)*length(Comp.train[[m]])))*100,"%\n", sep = "")
      
      k <- k + 1
    }
  }
  
  {
    CEEMD3.Metrics <- matrix(nrow = dim(combs)[1],ncol = 4)
    CEEMD3.Metrics.train <- matrix(nrow = dim(combs)[1],ncol = 4)
    colnames(CEEMD3.Metrics)       <- c("i","MAE","MAPE","RMSE")
    colnames(CEEMD3.Metrics.train) <- c("i","MAE","MAPE","RMSE")
    rownames(CEEMD3.Metrics)       <- model.list
    rownames(CEEMD3.Metrics.train) <- model.list
    
    
    CEEMD3.Pred.train <- matrix(nrow = dim(IMF.xtrain[[1]])[1], ncol = dim(combs)[1])
    CEEMD3.Pred.test  <- matrix(nrow = dim(IMF.xtest[[1]])[1], ncol = dim(combs)[1])
    CEEMD3.Prediction <- matrix(nrow = n, ncol = dim(combs)[1])
  }
  
  for (i in 1:dim(combs)[1]) {
    CEEMD3.Pred.train[,i] <- (PTRmo[[combs[i,1]]][,1] + 
                                PTRmo[[combs[i,2]]][,2] + 
                                PTRmo[[combs[i,3]]][,3] +
                                PTRmo[[combs[i,4]]][,4] + 
                                PTRmo[[combs[i,5]]][,5] +
                                PTRmo[[combs[i,6]]][,6])
    
    CEEMD3.Pred.test[,i] <- (PTEmo[[combs[i,1]]][,1] + 
                               PTEmo[[combs[i,2]]][,2] + 
                               PTEmo[[combs[i,3]]][,3] +
                               PTEmo[[combs[i,4]]][,4] + 
                               PTEmo[[combs[i,5]]][,5] +
                               PTEmo[[combs[i,6]]][,6])
    
    ### Avoiding negative values
    for (j in 1:dim(CEEMD3.Pred.train)[1]) {
      if (CEEMD3.Pred.train[j,i] < 0) {
        CEEMD3.Pred.train[j,i] <- 0
      }
    }
    for (j in 1:dim(CEEMD3.Pred.test)[1]) {
      if (CEEMD3.Pred.test[j,i] < 0) {
        CEEMD3.Pred.test[j,i] <- 0
      }
    }
    
    CEEMD3.Prediction[,i] <- c(CEEMD3.Pred.train[,i],CEEMD3.Pred.test[,i])
    
    # #Metrics
    CEEMD3.MAE  <- MAE(CEEMD3.Pred.test[,i], Obs.test)
    CEEMD3.MAPE <- mape(CEEMD3.Pred.test[,i], Obs.test)
    CEEMD3.RMSE <- RMSE(CEEMD3.Pred.test[,i], Obs.test)
    
    CEEMD3.MAE.train  <- MAE(CEEMD3.Pred.train[,i], Obs.train)
    CEEMD3.MAPE.train <- mape(CEEMD3.Pred.train[,i], Obs.train)
    CEEMD3.RMSE.train <- RMSE(CEEMD3.Pred.train[,i], Obs.train)
    
    CEEMD3.Metrics.train[i,] <- c(i,
                                  CEEMD3.MAE.train,
                                  CEEMD3.MAPE.train,
                                  CEEMD3.RMSE.train)
    CEEMD3.Metrics[i,] <- c(i,
                            CEEMD3.MAE,
                            CEEMD3.MAPE,
                            CEEMD3.RMSE)
  }
  
  xtable::xtable(CEEMD3.Metrics, digits = 4)
  
  save.image("3steps-CEEMD-results.RData")
}

# CEEMD-Stack Steps-ahead Prediction ----

## 2-step-ahead 
{
  # setwd(ResultsDir)
  # load('2steps-CEEMD-results.RData')
  
  set.seed(1234)
  stack2.database <- data.frame(Obs,CEEMD2.Prediction)
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
    obsTR <- matrix(seq(1,length(Obs.train),1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,length(Obs.test),1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.train))
    PTEmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.test))
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
          if(p==obsTR[v,1]) {
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
          if(p==obsTE[v,1]) {
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
    
    pred2.MAE.train  <- MAE(PTRmo[,m], Obs.train)
    pred2.MAPE.train <- mape(PTRmo[,m], Obs.train)
    pred2.RMSE.train <- RMSE(PTRmo[,m], Obs.train)
    
    pred2.MAE  <- MAE(PTEmo[,m], Obs.test)
    pred2.MAPE <- mape(PTEmo[,m], Obs.test)
    pred2.RMSE <- RMSE(PTEmo[,m], Obs.test)
    
    
    stack2.Metrics.train[m,] <- c(m,pred2.MAE.train,pred2.MAPE.train,pred2.RMSE.train)
    stack2.Metrics[m,] <- c(m,pred2.MAE,pred2.MAPE,pred2.RMSE)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
    
    save.image("2step-CEEMD-Stack-results.RData")
  }
  
  xtable::xtable(stack2.Metrics, digits = 4)
  
}

## 3-step-ahead 
{
  # setwd(ResultsDir)
  # load('3steps-CEEMD-results.RData')
  
  set.seed(1234)
  stack3.database <- data.frame(Obs,CEEMD3.Prediction)
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
    obsTR <- matrix(seq(1,length(Obs.train)+1,1),ncol = h, byrow = TRUE)
    obsTE <- matrix(seq(1,length(Obs.test),1),ncol = h, byrow = TRUE)
    PTRmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.train))
    PTEmo <- matrix(ncol = length(preprocess.list), nrow = length(Obs.test))
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
          if(p==obsTR[v,1]) {
            PTRmo[p,m] <- predict(stack[[m]],x_trainm[p,])
          }
          else if(p==obsTR[v,2]) {
            x_trainm[p,1] <- PTRmo[p-1,m]
            PTRmo[p,m]    <- predict(stack[[m]],x_trainm[p,])
          }
          else {
            x_trainm[p,1] <- PTRmo[p-1,m]
            x_trainm[p,2] <- x_trainm[p-1,1]
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
          if(p==obsTE[v,1]) {
            PTEmo[p,m] <- predict(stack[[m]],x_testm[p,])
          }
          else if(p==obsTE[v,2]) {
            x_testm[p,1] <- PTEmo[p-1,m]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
          else {
            x_testm[p,1] <- PTEmo[p-1,m]
            x_testm[p,2] <- x_testm[p-1,1]
            PTEmo[p,m]    <- predict(stack[[m]],x_testm[p,])
          }
        }
        else {
          break
        }
      }
    }
    
    stack3.pred[,m] <- c(PTRmo[,m],PTEmo[,m])
    
    ### Metrics
    
    pred3.MAE.train  <- MAE(PTRmo[,m], Obs.train)
    pred3.MAPE.train <- mape(PTRmo[,m], Obs.train)
    pred3.RMSE.train <- RMSE(PTRmo[,m], Obs.train)
    
    pred3.MAE  <- MAE(PTEmo[,m], Obs.test)
    pred3.MAPE <- mape(PTEmo[,m], Obs.test)
    pred3.RMSE <- RMSE(PTEmo[,m], Obs.test)
    
    
    stack3.Metrics.train[m,] <- c(m,pred3.MAE.train,pred3.MAPE.train,pred3.RMSE.train)
    stack3.Metrics[m,] <- c(m,pred3.MAE,pred3.MAPE,pred3.RMSE)
    
    cat("stack: ", preprocess.list[m]," \t",m/length(preprocess.list)*100, "%\n", sep = "")
    
    save.image("3step-CEEMD-Stack-results.RData")
  }
  
  xtable::xtable(stack3.Metrics, digits = 4)
  
}

# Diebold-Mariano tests ----
setwd(ResultsDir)

## One step ahead predictions dataframes
{
  pred_test_1 <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(pred_test_1) <- c('CEEMD--CORR--STACK',
                             'CEEMD--PCA--STACK',
                             'CEEMD--BC--STACK',
                             'CEEMD--KNN',
                             'CEEMD--SVRR',
                             'CEEMD--PLS',
                             'CEEMD--RF',
                             'CORR--STACK',
                             'PCA--STACK',
                             'BC--STACK',
                             'KNN',
                             'SVRR',
                             'PLS',
                             'RF')
  ## 1-3
  for (i in seq(3)) {
    pred_test_1[,i] <- tail(stack.pred[,i+6],n-cut)
  }
  ## 4-7
  for (i in seq(4)) {
    pred_test_1[,i+3] <- tail(CEEMD.Prediction[,i], n-cut)
  }
  load('stack-results.RData')
  ## 8-10
  for (i in seq(3)) {
    pred_test_1[,i+7] <- tail(stack.pred[,i+6],n-cut)
  }
  # 11-14
  for (i in seq(4)) {
    pred_test_1[,i+10] <- tail(pred[,i], n-cut)
  }
  
  save(pred_test_1, file = 'pred-day3-10minutes.RData')  
}

## Two step ahead predictions dataframes
{
  pred_test_2 <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(pred_test_2) <- c('CEEMD--CORR--STACK',
                             'CEEMD--PCA--STACK',
                             'CEEMD--BC--STACK',
                             'CEEMD--KNN',
                             'CEEMD--SVRR',
                             'CEEMD--PLS',
                             'CEEMD--RF',
                             'CORR--STACK',
                             'PCA--STACK',
                             'BC--STACK',
                             'KNN',
                             'SVRR',
                             'PLS',
                             'RF')
  ## 1-3
  for (i in seq(3)) {
    pred_test_2[,i] <- tail(stack2.pred[,i],n-cut)
  }
  ## 4-7
  for (i in seq(4)) {
    pred_test_2[,i+3] <- tail(CEEMD2.Prediction[,i], n-cut)
  }
  load('2steps-stack-results.RData')
  ## 8-10
  for (i in seq(3)) {
    pred_test_2[,i+7] <- tail(stack2.pred[,i],n-cut)
  }
  # 11-14
  for (i in seq(4)) {
    pred_test_2[,i+10] <- tail(Pred2step[,i], n-cut)
  }
  
  save(pred_test_2, file = 'pred-day3-20minutes.RData')  
}

## Three steps ahead predictions dataframes
{
  pred_test_3 <- matrix(ncol = 14, nrow = length(Obs.test))
  colnames(pred_test_3) <- c('CEEMD--CORR--STACK',
                             'CEEMD--PCA--STACK',
                             'CEEMD--BC--STACK',
                             'CEEMD--KNN',
                             'CEEMD--SVRR',
                             'CEEMD--PLS',
                             'CEEMD--RF',
                             'CORR--STACK',
                             'PCA--STACK',
                             'BC--STACK',
                             'KNN',
                             'SVRR',
                             'PLS',
                             'RF')
  ## 1-3
  for (i in seq(3)) {
    pred_test_3[,i] <- tail(stack3.pred[,i],n-cut)
  }
  ## 4-7
  for (i in seq(4)) {
    pred_test_3[,i+3] <- tail(CEEMD3.Prediction[,i], n-cut)
  }
  load('3steps-stack-results.RData')
  ## 8-10
  for (i in seq(3)) {
    pred_test_3[,i+7] <- tail(stack3.pred[,i],n-cut)
  }
  # 11-14
  for (i in seq(4)) {
    pred_test_3[,i+10] <- tail(Pred3step[,i], n-cut)
  }
  
  save(pred_test_3, file = 'pred-day3-30minutes.RData')  
}

## DM tests
### 1 step
{
  h <- 1
  
  e <- matrix(ncol = 14, nrow = n-cut)
  colnames(e) <- c('CEEMD--CORR--STACK',
                   'CEEMD--PCA--STACK',
                   'CEEMD--BC--STACK',
                   'CEEMD--KNN',
                   'CEEMD--SVRR',
                   'CEEMD--PLS',
                   'CEEMD--RF',
                   'CORR--STACK',
                   'PCA--STACK',
                   'BC--STACK',
                   'KNN',
                   'SVRR',
                   'PLS',
                   'RF')
  
  for (i in seq(dim(e)[2])) {
    e[,i] <- (Obs.test - pred_test_1[,i])
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.presult <- matrix(nrow = dim(e)[2], ncol = 1)
  colnames(DM.tvalue) <- 'DM.t-value'
  rownames(DM.tvalue) <- colnames(e)
  colnames(DM.pvalue) <- 'DM.p-value'
  rownames(DM.pvalue) <- colnames(e)
  colnames(DM.presult) <- 'DM.p-value'
  rownames(DM.presult) <- colnames(e)
  
  for (i in 1) {
    for (j in seq(dim(e)[2])) {
      if(i>=j) {
        DM.tvalue[j,i] <- NA
        DM.pvalue[j,i] <- NA
      }
      else {
        DMtest <- dm.test(e[,i],e[,j], h = h, power = 1)
        DM.tvalue[j,i] <- DMtest$statistic
        DM.pvalue[j,i] <- DMtest$p.value
        
        if(DM.pvalue[j,i] < 0.01){
          DM.presult[j,i] <- 'less than 0.01'
        }
        else if(DM.pvalue[j,i] < 0.05 && DM.pvalue[j,i]>0.01){
          DM.presult[j,i] <- 'less than 0.05'
        }
        else if(DM.pvalue[j,i] > 0.05){
          DM.presult[j,i] <- 'greater than 0.05'
        }
      }
    }
  }
  xtable::xtable(DM.tvalue, digits = 4)
  xtable::xtable(DM.pvalue, digits = 4)
  DM.presult
}
### 2 steps
{
  h <- 2
  
  e <- matrix(ncol = 14, nrow = n-cut)
  colnames(e) <- c('CEEMD--CORR--STACK',
                   'CEEMD--PCA--STACK',
                   'CEEMD--BC--STACK',
                   'CEEMD--KNN',
                   'CEEMD--SVRR',
                   'CEEMD--PLS',
                   'CEEMD--RF',
                   'CORR--STACK',
                   'PCA--STACK',
                   'BC--STACK',
                   'KNN',
                   'SVRR',
                   'PLS',
                   'RF')
  
  for (i in seq(dim(e)[2])) {
    e[,i] <- (Obs.test - pred_test_2[,i])
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.presult <- matrix(nrow = dim(e)[2], ncol = 1)
  colnames(DM.tvalue) <- 'DM.t-value'
  rownames(DM.tvalue) <- colnames(e)
  colnames(DM.pvalue) <- 'DM.p-value'
  rownames(DM.pvalue) <- colnames(e)
  colnames(DM.presult) <- 'DM.p-value'
  rownames(DM.presult) <- colnames(e)
  
  for (i in 1) {
    for (j in seq(dim(e)[2])) {
      if(i>=j) {
        DM.tvalue[j,i] <- NA
        DM.pvalue[j,i] <- NA
      }
      else {
        DMtest <- dm.test(e[,i],e[,j], h = h, power = 1)
        DM.tvalue[j,i] <- DMtest$statistic
        DM.pvalue[j,i] <- DMtest$p.value
        
        if(DM.pvalue[j,i] < 0.01){
          DM.presult[j,i] <- 'p-value less than 0.01'
        }
        else if(DM.pvalue[j,i] < 0.05 && DM.pvalue[j,i]>0.01){
          DM.presult[j,i] <- 'p-value less than 0.05'
        }
        else if(DM.pvalue[j,i] > 0.05){
          DM.presult[j,i] <- 'p-value greater than 0.05'
        }
      }
    }
  }
  xtable::xtable(DM.tvalue, digits = 4)
  xtable::xtable(DM.pvalue, digits = 4)
  DM.presult
}
### 3 steps
{
  h <- 3
  
  e <- matrix(ncol = 14, nrow = n-cut)
  colnames(e) <- c('CEEMD--CORR--STACK',
                   'CEEMD--PCA--STACK',
                   'CEEMD--BC--STACK',
                   'CEEMD--KNN',
                   'CEEMD--SVRR',
                   'CEEMD--PLS',
                   'CEEMD--RF',
                   'CORR--STACK',
                   'PCA--STACK',
                   'BC--STACK',
                   'KNN',
                   'SVRR',
                   'PLS',
                   'RF')
  
  for (i in seq(dim(e)[2])) {
    e[,i] <- (Obs.test - pred_test_3[,i])
  }
  
  DM.tvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.pvalue <- matrix(nrow = dim(e)[2], ncol = 1)
  DM.presult <- matrix(nrow = dim(e)[2], ncol = 1)
  colnames(DM.tvalue) <- 'DM.t-value'
  rownames(DM.tvalue) <- colnames(e)
  colnames(DM.pvalue) <- 'DM.p-value'
  rownames(DM.pvalue) <- colnames(e)
  colnames(DM.presult) <- 'DM.p-value'
  rownames(DM.presult) <- colnames(e)
  
  for (i in 1) {
    for (j in seq(dim(e)[2])) {
      if(i>=j) {
        DM.tvalue[j,i] <- NA
        DM.pvalue[j,i] <- NA
      }
      else {
        DMtest <- dm.test(e[,i],e[,j], h = h, power = 1)
        DM.tvalue[j,i] <- DMtest$statistic
        DM.pvalue[j,i] <- DMtest$p.value
        
        if(DM.pvalue[j,i] < 0.01){
          DM.presult[j,i] <- 'p-value less than 0.01'
        }
        else if(DM.pvalue[j,i] < 0.05 && DM.pvalue[j,i]>0.01){
          DM.presult[j,i] <- 'p-value less than 0.05'
        }
        else if(DM.pvalue[j,i] > 0.05){
          DM.presult[j,i] <- 'p-value greater than 0.05'
        }
      }
    }
  }
  xtable::xtable(DM.tvalue, digits = 4)
  xtable::xtable(DM.pvalue, digits = 4)
  DM.presult
}

