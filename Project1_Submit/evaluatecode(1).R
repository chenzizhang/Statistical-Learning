# Assume the Y value for the test data is stored in a two-column 
# data frame named "test.y":
# col 1: PID
# col 2: Sale_Price
#code to test if a specific model satisfies the criteria. will output a vector showing which split fails
#it's more strict than the instructor's requirement. Because the instructor requires that for each split, as long as one model success, then you success.
#since each of us take over one model separately, we can only test one model. Hope our model successes consistently
#we could modified it on sunday to make it test for three models at the same time
#my poor english, ni neng kan dong ba?
load("project1_testIDs.R")
data <- read.csv("Ames_data.csv")
firstfivecriteria <- 0.125 #criteria applied to the first five split
secondfivecriteria <- 0.135 #criteria applied to the second five splits
ifsuc <- rep("SUC",10)
rate <- rep(0,10)
rate2 <- rep(0,10)
ifsuc2 <- rep("SUC",10)
rate3 <- rep(0,10)
ifsuc3 <- rep("SUC",10)
for (j in 1:10){
  train <- data[-testIDs[,j], ]
  test <- data[testIDs[,j], ]
  test.y <- test[, c(1, 83)]
  test <- test[, -83]
  write.csv(train,"train.csv",row.names=FALSE)
  write.csv(test, "test.csv",row.names=FALSE)
  write.csv(test.y, "test_y.csv",row.names=FALSE)
  source('mymain.R')#your model here.make sure your model outputs "mysubmission.txt")
pred <- read.csv("mysubmission1.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
performance <- sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
rate[j] <- performance
if (j <= 5){
  if (performance >=firstfivecriteria){
  ifsuc[j] <- "FAIL"}
}else{
  if (performance >=secondfivecriteria){
    ifsuc[j] <- "FAIL"
  }
}
pred <- read.csv("mysubmission2.txt")
pred <- merge(pred, test.y, by="PID")
performance <- sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
rate2[j] <- performance
if (j <= 5){
  if (performance >=firstfivecriteria){
    ifsuc2[j] <- "FAIL"}
}else{
  if (performance >=secondfivecriteria){
    ifsuc2[j] <- "FAIL"
  }
}
pred <- read.csv("mysubmission3.txt")
names(test.y)[2] <- "True_Sale_Price"
pred <- merge(pred, test.y, by="PID")
performance <- sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
rate3[j] <- performance
if (j <= 5){
  if (performance >=firstfivecriteria){
    ifsuc3[j] <- "FAIL"}
}else{
  if (performance >=secondfivecriteria){
    ifsuc3[j] <- "FAIL"
  }
}
}
result <- data.frame(rate=rate,ifsuc=ifsuc,rate2=rate2,ifsuc2=ifsuc2,rate3=rate3,issuc3=ifsuc3)
write.csv(result,"finalresult.csv",row.names = FALSE)
