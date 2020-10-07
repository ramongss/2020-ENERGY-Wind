# Set environment ---------------------------------------------------------
rm(list = ls())
Sys.setenv("LANGUAGE" = "En")
Sys.setlocale("LC_ALL", "English")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# save several directories
BaseDir       <- getwd()
CodesDir      <- paste(BaseDir, "Codes", sep = "/")
FiguresDir    <- paste(BaseDir, "Figures", sep = "/")
ResultsDir    <- paste(BaseDir, "Results", sep = "/")
DataDir       <- paste(BaseDir, "Data",sep = "/")

# load Packages
setwd(CodesDir)
source("checkpackages.R")
source("ceemd_stack_pred.R")
source("stack_pred.R")

packages <- c('vmd','dplyr','tidyverse','magrittr', 'caret', 'reshape2', 'gghighlight',
              'TTR', 'forecast', 'Metrics', 'e1071', "cowplot", "elmNNRcpp", "hht", "grid",
              "tcltk", "foreach", "iterators","doParallel","lmtest","magrittr", "ggpubr")

sapply(packages,packs)

rm(packages)

library(extrafont)
# font_import(pattern = 'CM')
library(ggplot2)
library(Cairo)

# Data treatment ----------------------------------------------------------
# set working directory
setwd(DataDir) 

# load data
raw_data <- readxl::read_excel('dataset1.xlsx')

# format date column
raw_data$PCTimeStamp <- raw_data$PCTimeStamp %>% as.Date()

# drop na
raw_data <- raw_data %>% tidyr::drop_na()

# set a list of dates to analyze
months <- c(8,9,10)
dates <- paste0('2017-',months,'-01') %>% as.Date() %>% format("%Y-%m")

# create empty lists
wind_data <- list()
ceemd_stack_results <- list()
stack_results <- list()

# list of models
model_list <- c(
  'ridge',
  'svmLinear2',
  'knn',
  'pls'
) %>% sort()

for (month in seq(months)) {
  # filtering the data according to month list
  wind_data[[month]] <- raw_data[months(raw_data$PCTimeStamp) %in% month.name[months[month]],] %>% as.data.frame()
  
  # training using ceemd stack
  ceemd_stack_results[[month]] <- ceemd_stack_pred(wind_data[[month]], model_list)

  # training using stack models
  stack_results[[month]] <- stack_pred(wind_data[[month]], model_list)
}

# Save results ------------------------------------------------------------
# set working directory
setwd(ResultsDir)

# loop to save RDS
for (dataset in seq(dates)) {
  saveRDS(
    object = ceemd_stack_results[[dataset]],
    file = paste0('results_',dates[dataset],'_ceemd_stack.rds')
  )
  
  saveRDS(
    object = stack_results[[dataset]],
    file = paste0('results_',dates[dataset],'_stack.rds')
  )
}

# loop to save metrics results
FH <- c('One-steps','Two-steps','Three-steps') # aux to create forecasting horizon column

for (dataset in seq(dates)) {
  filename_ceemd_stack <- paste0('dataset_',dates[dataset],'_ceemd_stack_metrics.csv') # ceemd stack file name
  filename_stack <- paste0('dataset_',dates[dataset],'_stack_metrics.csv') # stack file name
  
  file.create(filename_ceemd_stack) # create file ceemd
  file.create(filename_stack) # create file stack
  
  # append header in csv files
  data.frame('model','FH','MAE','MAPE','RMSE') %>%
    write.table(file = filename_ceemd_stack,
                append = TRUE,
                sep = ',',
                col.names = FALSE,
                row.names = FALSE)
  data.frame('model','FH','MAE','MAPE','RMSE') %>%
    write.table(file = filename_stack,
                append = TRUE,
                sep = ',',
                col.names = FALSE,
                row.names = FALSE)
  
  for (metric in seq(ceemd_stack_results[[dataset]]$CEEMD_Metrics)) {
    # save ceemd_stack metrics in csv
    data.frame(
      FH = rep(FH[metric]),
      ceemd_stack_results[[dataset]]$STACK_Metrics[[metric]][,-1]
    ) %>%
      write.table(file = filename_ceemd_stack,
                  append = TRUE,
                  sep = ',',
                  col.names = FALSE,
                  row.names = TRUE)
    # save ceemd metrics in csv
    data.frame(
      FH = rep(FH[metric]),
      ceemd_stack_results[[dataset]]$CEEMD_Metrics[[metric]][,-1]
    ) %>%
      write.table(file = filename_ceemd_stack,
                  append = TRUE,
                  sep = ',',
                  col.names = FALSE,
                  row.names = TRUE)
    
    # save stack metrics in csv
    data.frame(
      FH = rep(FH[metric]),
      stack_results[[dataset]]$STACK_Metrics[[metric]][,-1]
    ) %>%
      write.table(file = filename_stack,
                  append = TRUE,
                  sep = ',',
                  col.names = FALSE,
                  row.names = TRUE)
    # save single metrics in csv
    data.frame(
      FH = rep(FH[metric]),
      stack_results[[dataset]]$Metrics[[metric]][,-1]
    ) %>%
      write.table(file = filename_stack,
                  append = TRUE,
                  sep = ',',
                  col.names = FALSE,
                  row.names = TRUE)
  }
}

# Load data --------------------------------------------------------------------
setwd(ResultsDir)

## load data
file_list <- list.files(pattern = '.rds') # list the .rds files

ceemd_stack_results <- list()
stack_results <- list()

count_stack <- 1 # single models aux counter
count_ceemd_stack <- 1 # vmd models aux counter 

# loop to read data
for (dataset in seq(file_list)) {
  if (dataset %% 2 != 0) {
    ceemd_stack_results[[count_ceemd_stack]] <- readRDS(file = file_list[dataset])
    count_ceemd_stack <- count_ceemd_stack + 1
  } else {
    stack_results[[count_stack]] <- readRDS(file = file_list[dataset])
    count_stack <- count_stack + 1
  }
}

## Plot Predict x Observed test ----
setwd(FiguresDir)
datasets <- list()
datasets24h <- list()
# datasets for test sets
for (ii in seq(3)) {
  if (ii == 1){
    datasets[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 1008),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'corr'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 1008),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'BoxCox'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 1008),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 1008)
    ) %>% melt() %>% data.frame(
      rep(seq(1008)),
      .,
      rep(c('Observed','Predicted'), each = 1008),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*1008)
    )
  } else if (ii == 2) {
    datasets[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 1008),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'corr'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 1008),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'corr'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 1008),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 1008)
    ) %>% melt() %>% data.frame(
      rep(seq(1008)),
      .,
      rep(c('Observed','Predicted'), each = 1008),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*1008)
    )
  } else {
    datasets[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 1008),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'BoxCox'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 1008),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'BoxCox'], 1008),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 1008),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 1008)
    ) %>% melt() %>% data.frame(
      rep(seq(1008)),
      .,
      rep(c('Observed','Predicted'), each = 1008),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*1008)
    )
  }
  datasets[[ii]]$variable <- NULL
  datasets[[ii]]$Set <- rep('Last 7 days')
  colnames(datasets[[ii]]) <- c('x','value', 'type', 'FH', "Set")
}

# datasets for 24h sets
for (ii in seq(3)) {
  if (ii == 1){
    datasets24h[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 144),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'corr'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 144),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'BoxCox'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 144),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 144)
    ) %>% melt() %>% data.frame(
      rep(seq(144)),
      .,
      rep(c('Observed','Predicted'), each = 144),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*144)
    )
  } else if (ii == 2) {
    datasets24h[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 144),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'corr'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 144),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'corr'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 144),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 144)
    ) %>% melt() %>% data.frame(
      rep(seq(144)),
      .,
      rep(c('Observed','Predicted'), each = 144),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*144)
    )
  } else {
    datasets24h[[ii]] <- data.frame(
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'Obs'], 144),
      'OSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`1-step`[,'BoxCox'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'Obs'], 144),
      'TSA'      = tail(ceemd_stack_results[[ii]]$Predictions$`2-step`[,'BoxCox'], 144),
      'Observed' = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'Obs'], 144),
      'THA'      = tail(ceemd_stack_results[[ii]]$Predictions$`3-step`[,'pca'], 144)
    ) %>% melt() %>% data.frame(
      rep(seq(144)),
      .,
      rep(c('Observed','Predicted'), each = 144),
      rep(c("10 minutes","20 minutes","30 minutes"), each = 2*144)
    )
  }
  
  datasets24h[[ii]]$variable <- NULL
  datasets24h[[ii]]$Set <- rep('Last 24 hours')
  colnames(datasets24h[[ii]]) <- c('x','value', 'type', 'FH','Set')
}

# grid
for (dataset in seq(datasets)) {
  
  final_dataset <- rbind(datasets[[dataset]], datasets24h[[dataset]])
  
  # plot test set ----
  final_dataset$FH <- final_dataset$FH %>% factor(levels = c("10 minutes","20 minutes","30 minutes"))
  final_dataset$Set <- final_dataset$Set %>% factor(levels = c("Last 7 days", "Last 24 hours"))
  
  plot_test <- final_dataset %>% as.data.frame %>% 
    ggplot(aes(x = x, y = value, colour = type)) +
    geom_line(size = 1) +
    geom_rect(
      data = data.frame(Set = factor("Last 7 days")),
      aes(xmin = 864,
          xmax = 1020,
          ymin = -Inf,
          ymax = Inf),
      fill = "#FF7F00", alpha = .3, inherit.aes = F
    ) +
    geom_text(
      data = data.frame(
        Set = factor("Last 7 days"), 
        FH = factor("10 minutes")
      ),
      aes(x = 936, y = Inf, label = "Last 24 hours", family = "CM Roman"),
      vjust = -0.5,
      inherit.aes = F
    ) +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = 'bottom',
          legend.background = element_blank(),
          legend.text = element_text(size = 20),
          text = element_text(family = "CM Roman", size = 20),
          strip.placement = "outside",
          strip.background = element_blank(),
          panel.grid.minor = element_blank(),
    ) +
    ylab('Wind Power (KW)') +
    xlab('Test samples (10 minutes)') +
    facet_grid(rows = vars(FH), cols = vars(Set),scales = "free") +
    scale_color_manual(values = c("#377EB8","#E41A1C")) +
    coord_cartesian(clip = "off") 
  
  plot_test
  
  plot_test %>% 
    ggsave(
      filename = paste0('PO_',dates[dataset],'.pdf'),
      device = 'pdf',
      width = 12,
      height = 6.75,
      units = "in",
      dpi = 300
    )
  
}

## Plot IMFs ----
setwd(FiguresDir)

IMFs <- data.frame()
Aux <- data.frame()

for (dataset in seq(ceemd_stack_results)) {
  Aux <- ceemd_stack_results[[dataset]]$IMF
  Aux$dataset <- rep(dates[dataset])
  Aux$n <- seq(nrow(Aux))
  IMFs <- rbind(IMFs,Aux %>% melt(id.vars = c('dataset','n')))
}

IMFs$dataset <- IMFs$dataset %>% 
  factor(levels = c('2017-08','2017-09','2017-10'),
         labels = c(month.name[8:10]))

imf_labels <- 
  c(
    expression(paste(IMF[1])),
    expression(paste(IMF[2])),
    expression(paste(IMF[3])),
    expression(paste(IMF[4])),
    expression(paste(IMF[5])),
    'Residue'
  )

IMFs$variable <- IMFs$variable %>% 
  factor(
    levels = c('Obs', paste0('IMF',seq(5)), 'Residue'),
    labels = c('Obs',imf_labels)
  )

imf_plot <- IMFs %>% 
  filter(variable != 'Obs') %>%
  ggplot(aes(x = n, y = value, colour = variable)) +
  geom_line(size = 0.3, colour = '#377EB8') +
  theme_bw() +
  theme(
    text = element_text(family = "CM Roman", size = 16),
    strip.placement = "outside",
    strip.background = element_blank(),
    panel.grid.minor = element_blank(),
  ) +
  ylab('') + xlab('Samples(10 minutes)') +
  facet_grid(
    variable ~ dataset,
    scales = 'free',
    switch = 'y',
    labeller = "label_parsed",
  )

# imf_plot

imf_plot %>% 
  ggsave(
    filename = 'imf_plot.pdf',
    device = 'pdf',
    width = 12,
    height = 6.75,
    units = "in",
    dpi = 300
  ) 

## Plot datasets ----
setwd(FiguresDir)

obs_dataset <- data.frame()
Aux2 <- data.frame()

for (dataset in seq(wind_data)) {
  Aux2 <- data.frame(obs = wind_data[[dataset]][,'Power'])
  Aux2$n <- seq(nrow(wind_data[[dataset]]))
  Aux2$type <- c(rep('Training', times = nrow(wind_data[[dataset]])-1008), 
                        rep('Test', times = 1008))
  Aux2$dataset <- rep(paste0('dataset', dataset))
  obs_dataset <- rbind(obs_dataset, Aux2)
}

obs_dataset$dataset <- obs_dataset$dataset %>% 
  factor(
    levels = paste0('dataset', seq(wind_data)),
    labels = c(month.name[8:10])
  )

dataplot <- obs_dataset %>% 
  ggplot(aes(x = n, y = obs)) +
  geom_line(size = 0.5, colour = '#377EB8') +
  facet_grid(vars(dataset), scales = 'free', switch = 'y') +
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.position = 'bottom',
        legend.background = element_blank(),
        legend.text = element_text(size = 20),
        text = element_text(family = "CM Roman", size = 20),
        strip.placement = "outside",
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 20),
  ) +
  ylab('') + xlab('Samples (10 minutes)')

dataplot %>% 
  ggsave(
    filename = 'datasets_plot.pdf',
    device = 'pdf',
    width = 12,
    height = 6.75,
    units = "in",
    dpi = 300
  ) 

## Plot MAPE barplot ----
setwd(FiguresDir)

mape_metrics <- data.frame()
Aux3 <- data.frame()

for (dataset in seq(ceemd_stack_results)) {
  for (horizon in seq(months)) {
    Aux3 <- data.frame(mape = ceemd_stack_results[[dataset]]$STACK_Metrics[[horizon]][, "MAPE"]*100)
    Aux3$dataset <- rep(month.name[months[dataset]]) %>% as.factor()
    Aux3$model <- rep(c('BC', 'CORR', 'PCA'))
    Aux3$FH <- rep(paste0(horizon,'0 min'))
    rownames(Aux3) <- NULL
    mape_metrics <- rbind(mape_metrics, Aux3)
  }
}

mape_barplot <- mape_metrics %>% 
  ggplot(aes(x = FH, y = mape, fill = model)) +
  geom_bar(
    position = position_dodge(),
    stat = 'identity',
    color = 'black'
  ) +
  geom_text(
    aes(label = round(mape, 2)),
    position = position_dodge(width = 0.9),
    vjust = -0.25,
    family = 'CM Roman'
  ) +
  facet_wrap(vars(dataset), ncol = 3) +
  theme_bw() +
  theme(#legend.title = element_blank(),
        legend.position = 'bottom',
        legend.background = element_blank(),
        legend.text = element_text(size = 20),
        text = element_text(family = "CM Roman", size = 20),
        strip.placement = "outside",
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        strip.text = element_text(size = 20),
  ) +
  ylab('MAPE (%)') + xlab('Forecasting Horizon') + 
  scale_y_continuous(expand = c(0, NA), limits = c(0, 10)) +
  scale_fill_brewer(palette = 'Set1', name = 'Model')

mape_barplot %>% 
  ggsave(
    filename = 'mape_barplot.pdf',
    device = 'pdf',
    width = 12,
    height = 6.75,
    units = "in",
    dpi = 300
  ) 

## Plot errors' std radarplot ----
setwd(FiguresDir)

error <- list()
for (dataset in seq(ceemd_stack_results)) {
  error[[dataset]] <- data.frame()
  for (horizon in seq(ceemd_stack_results[[dataset]]$Errors)) {
    Aux4 <- tail(ceemd_stack_results[[dataset]]$Errors[[horizon]], 1008) %>% 
      apply(., 2, sd) %>% data.frame %>% t()
    Aux4 <- cbind(
      Aux4,
      tail(stack_results[[dataset]]$Errors[[horizon]], 1008) %>% 
        apply(., 2, sd) %>% data.frame %>% t()
    )
    
    error[[dataset]] <- rbind(error[[dataset]], Aux4)
  }
  rownames(error[[dataset]]) <- paste0(seq(3), '0 minutes')
  colnames(error[[dataset]]) <- LETTERS[1:14]
  
  error[[dataset]][,'F'] <- NULL 
  
  write.csv(
    x = error[[dataset]],
    file = paste0('dataset',dataset ,'_std.csv'),
    row.names = FALSE
  )
  
}
names(error) <- month.name[months]

## Summary table ----
summaries_table <- data.frame(
  'Variable' = rep(names(wind_data[[1]])[-1], times = 3),
  'Samples' = rep(c('Whole', 'Training', 'Test'), each = ncol(wind_data[[1]][-1]))
)

for (dataset in seq(length(wind_data))) {
  #Descriptives
  n <- nrow(wind_data[[dataset]])
  cut <- n - 1008
  
  #Whole
  Whole <- t(apply(wind_data[[dataset]][,-1],2,function(x){c(mean(x),sd(x),min(x),max(x))}))
  colnames(Whole) <- paste0(c('Mean.', 'Std.', 'Min.', 'Max.'), dataset)
  #Train Descriptives
  Train <- t(apply(wind_data[[dataset]][1:cut,-1],2,function(x){c(mean(x),sd(x),min(x),max(x))}))
  colnames(Train) <- names(Whole)
  #Test Descriptives
  Test <- t(apply(tail(wind_data[[dataset]][,-1],n - cut),2,function(x){c(mean(x),sd(x),min(x),max(x))}))
  colnames(Test) <- names(Whole)
  
  #Merge
  summaries_table <- cbind(summaries_table, rbind(Whole, Train, Test))
  row.names(summaries_table) <- NULL # reset row index
}

# Reorder rows
summaries_table <- summaries_table %>% 
  arrange(factor(Variable, levels = names(wind_data[[1]][-1])))

print(xtable::xtable(summaries_table, digits = 2), include.rownames = FALSE)

## DM test ----
error <- list()
DM_tvalue <- list()
DM_pvalue <- list()
DM_presult <- list()
for (dataset in seq(ceemd_stack_results)) {
  error[[dataset]] <- list()
  DM_tvalue[[dataset]] <- list()
  DM_pvalue[[dataset]] <- list()
  DM_presult[[dataset]] <- list()
  for (horizon in seq(ceemd_stack_results[[dataset]]$Errors)) {
    # taking errors from ceemd_stack
    error[[dataset]][[horizon]] <- tail(ceemd_stack_results[[dataset]]$Errors[[horizon]], 1008)
    # cbind the errors from ceemd_stack with stack
    error[[dataset]][[horizon]] <- cbind(error[[dataset]][[horizon]], tail(stack_results[[dataset]]$Errors[[horizon]], 1008))
    # rename columns
    colnames(error[[dataset]][[horizon]]) <- LETTERS[1:ncol(error[[dataset]][[horizon]])]
    
    # DM test
    DM_tvalue[[dataset]][[horizon]] <- matrix(
      nrow = ncol(error[[dataset]][[horizon]]),
      ncol = ncol(error[[dataset]][[horizon]])
    )
    colnames(DM_tvalue[[dataset]][[horizon]]) <- LETTERS[1:ncol(error[[dataset]][[horizon]])]
    rownames(DM_tvalue[[dataset]][[horizon]]) <- LETTERS[1:ncol(error[[dataset]][[horizon]])]
    DM_pvalue[[dataset]][[horizon]] <- DM_tvalue[[dataset]][[horizon]]
    DM_presult[[dataset]][[horizon]] <- DM_tvalue[[dataset]][[horizon]]
    
    for (col in seq(nrow(DM_tvalue[[dataset]][[horizon]]))) {
      for (row in seq(nrow(DM_tvalue[[dataset]][[horizon]]))) {
        if (col == row) {
          DM_tvalue[[dataset]][[horizon]][row, col] <- NA
          DM_pvalue[[dataset]][[horizon]][row, col] <- NA
        } else if (error[[dataset]][[horizon]][,col] == error[[dataset]][[horizon]][,row]) {
          DM_tvalue[[dataset]][[horizon]][row, col] <- NA
          DM_pvalue[[dataset]][[horizon]][row, col] <- NA
        } else {
          DMtest <- dm.test(
            error[[dataset]][[horizon]][,col],
            error[[dataset]][[horizon]][,row],
            h = horizon, power = 1
          )
          DM_tvalue[[dataset]][[horizon]][row,col] <- DMtest$statistic
          DM_pvalue[[dataset]][[horizon]][row,col] <- DMtest$p.value
          
          if (DM_pvalue[[dataset]][[horizon]][row,col] <= 0.01) {
            DM_presult[[dataset]][[horizon]][row,col] <- '*'
          } else if (DM_pvalue[[dataset]][[horizon]][row,col] <= 0.05 && DM_pvalue[[dataset]][[horizon]][row,col] > 0.01){
            DM_presult[[dataset]][[horizon]][row,col] <- '**'
          } else if (DM_pvalue[[dataset]][[horizon]][row,col] > 0.05){
            DM_presult[[dataset]][[horizon]][row,col] <- NA
          }
        }
      }
    }
    
  }
  names(error[[dataset]]) <- paste0(seq(3),'-step')
  names(DM_tvalue[[dataset]]) <- paste0(seq(3),'-step')
  names(DM_pvalue[[dataset]]) <- paste0(seq(3),'-step')
  names(DM_presult[[dataset]]) <- paste0(seq(3),'-step')
}
names(error) <- month.name[8:10]
names(DM_tvalue) <- month.name[8:10]
names(DM_pvalue) <- month.name[8:10]
names(DM_presult) <- month.name[8:10]
