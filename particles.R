
# Clear environment
rm(list = ls())

# Call Libraries
library(caret)
library(tidyverse)
library(data.table)
library(dplyr)
library(readr)
library(caret)
library(gradDescent)
library(tibble)
library(sgd)
library(xgboost)
library(ROCR)


#### `read.csv()`
datadir = "Data"
datafiles = list.files(path=datadir, pattern="*SY.csv", full.names=TRUE)
datafiles


# tbl_ben_mal <-   
#   list.files(path=datadir, pattern="*.csv", full.names=TRUE) %>% 
#   map_df(~fread(.))

# filenames <- list.files(path="Data",
#                         pattern=".*csv", full.names=TRUE)

# change working directory
setwd("./Data/")
getwd()

# file names
filenames <- gsub("\\.csv$","", list.files(pattern="\\SY.csv$"))

# Import Data 
for(i in filenames){
  assign(i, read.csv(header = FALSE, paste(i, ".csv", sep="")))
}

# change to parent directory
setwd("..")
getwd()


# Change the name of response variable
names(SUSY)[1] <- c("Particles")

# Move response variable to the end of the data set
SUSY <- data.frame(SUSY[, c(2:ncol(SUSY), 1)])

# benign_traffic$Attack <- 0
# 
# head(benign_traffic[ , (ncol(benign_traffic)-4):ncol(benign_traffic)])
# 
# 
# combo$Attack <- 1
# junk$Attack <- 1
# scan$Attack <- 1
# tcp$Attack <- 1
# udp$Attack <- 1
# 
# 
# head(combo[ , (ncol(combo)-4):ncol(combo)])
# head(junk[ , (ncol(junk)-4):ncol(junk)])
# head(scan[ , (ncol(scan)-4):ncol(scan)])
# head(tcp[ , (ncol(tcp)-4):ncol(tcp)])
# head(udp[ , (ncol(udp)-4):ncol(udp)])
# 
# 
# # Combine multple data frames to one
# dftraffic <- rbind(benign_traffic, combo, junk, scan, tcp, udp)
# 
# # Remove all objects except train data set
# rm(list= ls()[!(ls() %in% c('dftraffic'))])


# Divide train dataset in order to test the model's accuracy
ind <- createDataPartition(SUSY$Particles, p = 0.8, list=FALSE)
training <- SUSY[ind, ]
test <- SUSY[-ind, ]


# Remove starting data set for memory purposes
rm("SUSY")
