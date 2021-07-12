maxiter <- 1000


# Function to standardize input values
zscore <- function(x, mean.val=NA) {
  if(is.matrix(x)) return(apply(x, 2, zscore, mean.val=mean.val))
  if(is.data.frame(x)) return(data.frame(apply(x, 2, zscore, mean.val=mean.val)))
  if(is.na(mean.val)) mean.val <- mean(x)
  sd.val <- sd(x)
  if(all(sd.val == 0)) return(x) # if all the values are the same
  (x - mean.val) / sd.val 
}

# Standardize the features
training.scaled <- zscore(training[, 1:(ncol(training) - 1)])

# Add the response variable
training.scaled$Particles <- training$Particles

# Standardize the features
test.scaled <- zscore(test[, 1:(ncol(test) - 1)])

# Add the response variable
test.scaled$Particles <- test$Particles


# Gradient descent function
grad <- function(x, y, theta) {
  gradient <- (1 / nrow(y)) * (t(x) %*% (1/(1 + exp(-x %*% t(theta))) - y))
  return(t(gradient))
}

gradient.descent <- function(x, y, alpha=0.1, num.iterations, threshold=1e-3, output.path=FALSE) {
  
  # Add x_0 = 1 as the first column
  m <- if(is.vector(x)) length(x) else nrow(x)
  if(is.vector(x) || (!all(x[,1] == 1))) x <- cbind(rep(1, m), x)
  if(is.vector(y)) y <- matrix(y)
  x <- apply(x, 2, as.numeric)
  
  num.features <- ncol(x)
  
  # Initialize the parameters
  theta <- matrix(rep(0, num.features), nrow=1)
  
  # Look at the values over each iteration
  theta.path <- theta
  for (i in 1:num.iterations) {
    theta <- theta - alpha * grad(x, y, theta)
    if(all(is.na(theta))) break
    theta.path <- rbind(theta.path, theta)
    if(i > 2) if(all(abs(theta - theta.path[i-1,]) < threshold)) break 
  }
  
  if(output.path) return(theta.path) else return(theta.path[nrow(theta.path),])
}

set.seed(12)
start_time <- Sys.time()
unscaled.theta <- gradient.descent(x = training[ , 1:(ncol(training) - 1)], y = training$Particles, num.iterations = 1000, output.path=TRUE)
end_time <- Sys.time()
end_time - start_time

# scaled.theta <- gradient.descent(x = training.scaled, y = training$Particles, num.iterations = 1000, output.path=TRUE)


set.seed(13)
start_time_new <- Sys.time()
unscaled.theta.2 <- gradient.descent(x = training[ , 1:(ncol(training) - 1)], y = training$Particles, num.iterations = 2000, output.path=TRUE)
end_time_new <- Sys.time()
end_time_new - start_time_new


# Estimated y based on final values (optimal parameters) of unscaled theta
y <- c()
for (j in 1:ncol(unscaled.theta)) {
  for (i in 1:(nrow(test))) {
    y <- rbind(y,tibble(exp(unscaled.theta[1001, 1] +sum(test[i, 1:(ncol(test)-1)]*unscaled.theta[1001, 2:ncol(unscaled.theta)]))/
                          (1+exp(unscaled.theta[1001, 1] + sum(test[i, 1:(ncol(test)-1)]*unscaled.theta[1001, 2:ncol(unscaled.theta)])))))
  }
  return(y)
}

# Remove attribute names
names(y) <- NULL


#========================== Model Train - Logistic Regression ==================================================

# Logistic Regression model
start_time_glm <- Sys.time()
testmodel <- glm(Particles ~ ., data = training, family = binomial("logit"))
end_time_glm <- Sys.time()
end_time_glm - start_time_glm

summary(testmodel)

# Save logistic model
saveRDS(testmodel, file="log_regression.rds")

# log - likelihoods
log_likelihoods <- function(actuals, probs) {
  actuals <- as.numeric(actuals)
  probs <- as.numeric(probs)
  actuals * log(probs) + (1 - actuals) * log(1 - probs)
}

# dataset log - likehood
dataset_log_likelihood <- function(actuals, probs) {
  sum(log_likelihoods(actuals, probs))
}

# deviances
deviances <- function(actuals, probs) {
  -2 * log_likelihoods(actuals, probs)
}

# datatset deviance
dataset_deviance <- function(actuals, probs) {
  sum(deviances(actuals, probs))
}

# model deviance
model_deviance <- function(model, data, response) {
  actuals = data[[response]]
  probs = predict(model, newdata = data, type = "response")
  dataset_deviance(actuals, probs)
}

# Check our model's deviance (similar to summary output)
model_deviance(logmodel, data = training, response = "Particles")

# null deviance
null_deviance <- function(data, response) {
  actuals <- data[[response]]
  probs <- mean(data[[response]])
  dataset_deviance(actuals, probs)
}

# Check our model's null deviance (similar to summary output)
null_deviance(data = training, response = "Particles")

# Accuracy of the logistic gregression model - pseudo "R-Squared"
model_pseudo_r_squared <- function(model, data, response) {
  1 - ( model_deviance(model, data, response) /
          null_deviance(data, response) )
}

# Check our model's prediction accuracy (pseudo "R-Squared")
model_pseudo_r_squared(logmodel, data = training, response = "Particles")


# Check the significance of the deviance's difference
model_chi_squared_p_value <- function(model, data, response) {
  
  null_df <- nrow(data) - 1
  model_df <- nrow(data) - length(model$coefficients)
  difference_df <- null_df - model_df
  n_deviance <- null_deviance(data, response)
  m_deviance <- model_deviance(model, data, response)
  difference_deviance <- n_deviance - m_deviance
  pchisq(difference_deviance, difference_df,lower.tail = F)
  
}

# Chi - Squared p - value
model_chi_squared_p_value(logmodel, data = training, response = "Particles")



#========================== Model Train - XGBoost ==================================================

# Expand the grid of xgboost parameters
xgbGrid <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.1,
  gamma = 6,               
  colsample_bytree = 0.7,    
  min_child_weight = 0.8,    
  subsample = 0.7            
)

# Set the training control for xgboost
xgbTrControl <- trainControl(
  method = "repeatedcv",
  number = 4,
  repeats = 1,
  verboseIter = TRUE,
  returnData = FALSE,
  allowParallel = TRUE,
  classProbs = TRUE
)

# Model train
print("Train xgboost model")
start_time_sgd <- Sys.time()
xgbTrain <- train(
  x = as.matrix(training[, -which(names(training) %in% "Particles")]), 
  y = training$Particles,
  objective = "binary:logistic",
  trControl = xgbTrControl,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  eval_metric="logloss"
)
end_time_sgd <- Sys.time()
end_time_sgd - start_time_sgd


# Summary of the model
summary(xgbTrain)

# Optimal parameters of xgboost
xg.param <- list('objective' = 'binary:logistic',
                 'eval_metric' = 'error',
                 'eval_metric' = 'logloss',
                 'eta' = 0.1,
                 'gamma' = 6,
                 'max.depth' = 6,
                 'min_child_weight' = 0.8,
                 'subsample' = 0.7,
                 'colsample_bytree' = 0.7)#,
                 #'nthread' = 3)


# Xgboost train with cross validation
start_time_cv <- Sys.time()
xgb.fit.cv <- xgb.cv(param = xg.param, 
                     data = as.matrix(training[, -which(names(training) %in% "Particles")]), 
                     label = training$Particles, 
                     nfold = 4, nrounds = 100)
end_time_cv <- Sys.time()
end_time_cv - start_time_cv


# Save sgd model
saveRDS(xgb.fit.cv, file="sgd_cv.rds")

# Load xgboost_cv model
xgb.fit.cv <- readRDS("sgd_cv.rds")

# The minimum logloss in test data
min(xgb.fit.cv$evaluation_log$test_logloss_mean)

# The round where logloss reaches its optimal value (minimum)
cv.min.rounds <- which(xgb.fit.cv$evaluation_log$test_logloss_mean == min(xgb.fit.cv$evaluation_log$test_logloss_mean)) 
paste("The round where logloss reaches its optimal value (minimum) is:", cv.min.rounds)

# Plot logloss values to check the normality of the results (test df)
plot(xgb.fit.cv$evaluation_log$test_logloss_mean, ylab = "test data - logloss", ylim = c(0.4, 0.7))

# Plot logloss values to check the normality of the results (train df)
plot(xgb.fit.cv$evaluation_log$train_logloss_mean, ylab = "train data - logloss", ylim = c(0.4, 0.7))

# Optimal number of rounds
cv_rounds <- round(mean(which(xgb.fit.cv$evaluation_log$test_logloss_mean == min(xgb.fit.cv$evaluation_log$test_logloss_mean))))
cv_rounds                 

# Fit model on training set
start_time_xgb <- Sys.time()
xgb.fit <- xgboost(param = xg.param, 
                   data = as.matrix(training[, -which(names(training) %in% "Particles")]),
                   label = training$Particles,
                   nrounds = cv_rounds)
end_time_xgb <- Sys.time()
end_time_xgb - start_time_xgb

# Save model to binary local file 
xgb.save(xgb.fit, "xgboost2.model")

# Importance matrix of variables
importance_matrix <- xgb.importance(colnames(training[, -which(names(training) %in% "Particles")]), 
                                    model = xgb.fit)

xgb.plot.importance(importance_matrix, xlim = c(0, 0.4))

#========================== Models' Predictions =============================================

# Load logistic model
logmodel <- readRDS("log_regression.rds")

# Load xgboost_cv model
xgb.fit.cv <- readRDS("sgd_cv.rds")

# Load xgboost model
xgb.fit <- xgb.load("xgboost.model")

# Predictions of the trained logistic model in test data
logtest.predict <- predict(logmodel, newdata = test[, 1:(ncol(test)-1)], type = "response")

# Predictions of the trained sgd model in test data
sgd.predict <- predict(sgdmodel, newdata = as.matrix(test.scaled), type = "response")


# Predict with xgboost model 
xgb.probs <-predict(xgb.fit, 
                    newdata = as.matrix(test[, -which(names(test) %in% "Particles")]), 
                    type = "prob")

# First rows of predictions
head(xgb.probs, n = 10)

# Data frame with target and predicted variable
pred.target <- as.data.frame(cbind(test[, which(names(test) %in% "Particles")], xgb.probs))

# Change column names
colnames(pred.target) <- c("target", "prob")

# Summary statistics for data frame
summary(pred.target)

#Look at the confusion matrix  
confusionMatrix(as.factor(ifelse(xgb.probs > 0.5, 1, 0)), as.factor(pred.target$target))   

# Remove attribute names
names(logtest.predict) <- NULL

# Converting logistic probabilities to labels (set the cutoff to 0.5)
logtest.predictions <- as.numeric(logtest.predict >= 0.5)

# Converting gradient descent probabilities to labels (set the cutoff to 0.5)
gd.predictions <- as.numeric(y$y.estimated >= 0.5)

# Converting stochastic gradient descent probabilities to labels (set the cutoff to 0.5)
sgd.predictions <- as.numeric(sgd.predict >= 0.5)

# Converting xgboost probabilities to labels (set the cutoff to 0.5)
xgb.predictions <- as.numeric(pred.target$prob >= 0.5)

# Accuracy of the predicted and actual values in test data (gradient descent)
mean(gd.predictions == test$Particles)

# Accuracy of the predicted and actual values in test data (stochastic gradient descent)
mean(sgd.predictions == test$Particles)

# Accuracy of the predicted and actual values in test data (xgboost)
mean(xgb.predictions == test$Particles)

# Accuracy of the predicted and actual values in test data (logistic regression)
mean(logtest.predictions == test$Particles)

# Confusion Matrix logistic model
(confusion_matrix_lr <- table(predicted = logtest.predictions, actual = test$Particles))

# Confusion Matrix gradient descent model
(confusion_matrix_gd <- table(predicted = gd.predictions, actual = test$Particles))

# Confusion Matrix stochastic gradient descent model (xgboost)
(confusion_matrix_sgd <- table(predicted = xgb.predictions, actual = test$Particles))

# Accuracy of the predicted logistic model in test data
(accuracy_lr <- (confusion_matrix_lr[1, 1] + confusion_matrix_lr[2, 2]) / nrow(test))

# Accuracy of the predicted gradient descent model in test data
(accuracy_gd <- (confusion_matrix_gd[1, 1] + confusion_matrix_gd[2, 2]) / nrow(test))

# Accuracy of the predicted stochastic gradient descent model in test data
(accuracy_sgd <- (confusion_matrix_sgd[1, 1] + confusion_matrix_sgd[2, 2]) / nrow(test))


# Precision or Pos Pred Value of logistic model  in test data
(precision_lr <- confusion_matrix_lr[2, 2] / sum(confusion_matrix_lr[2, ]))

# Precision or Pos Pred Value of gradient descent model  in test data
(precision_gd <- confusion_matrix_gd[2, 2] / sum(confusion_matrix_gd[2, ]))

# Precision or Pos Pred Value of stochastic gradient descent model  in test data (xgboost)
(precision_sgd <- confusion_matrix_sgd[2, 2] / sum(confusion_matrix_sgd[2, ]))

# Sensitivity of logistic model in test data
(sensitivity_lr <- confusion_matrix_lr[2, 2] / sum(confusion_matrix_lr[, 2]))

# Sensitivity of gradient descent model in test data
(sensitivity_gd <- confusion_matrix_gd[2, 2] / sum(confusion_matrix_gd[, 2]))

# Sensitivity of stochastic gradient descent model in test data
(sensitivity_sgd <- confusion_matrix_sgd[2, 2] / sum(confusion_matrix_sgd[, 2]))

# Specificity of logistic regression model in test data
(specificity_lr <- confusion_matrix_lr[1, 1] / sum(confusion_matrix_lr[, 1]))

# Specificity of gradient descent model in test data
(specificity_gd <- confusion_matrix_gd[1, 1] / sum(confusion_matrix_gd[, 1]))

# Specificity of stochastic gradient descent model in test data
(specificity_sgd <- confusion_matrix_sgd[1, 1] / sum(confusion_matrix_sgd[, 1]))


#========================== ROC Curve ============================================================


# Dataframe with predictions (probs) + actual data (response)
pred <- prediction(xgb.probs, test$Particles)

# ROC Curve
perf <- performance(pred, measure = "prec", x.measure = "rec")

# Plot ROC Curve
plot(perf, xlab="Sensitivity")

# Thresholds or cutoffs
thresholds <- data.frame(cutoffs = perf@alpha.values[[1]], sensitivity =perf@x.values[[1]], precision = perf@y.values[[1]])

# Thresholds for sensitivity > 0.8 and precision > 0.7
subset(thresholds,(sensitivity > 0.7) & (precision > 0.8))

