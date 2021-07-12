
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

# Divide train dataset in order to test the model's accuracy
ind <- createDataPartition(SUSY$Particles, p = 0.8, list=FALSE)
training <- SUSY[ind, ]
test <- SUSY[-ind, ]


# Remove starting data set for memory purposes
rm("SUSY")
