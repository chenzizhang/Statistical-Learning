if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  "text2vec",
  "dplyr",
  "slam",
  "xgboost"
)
set.seed(101)
data <- read.csv("loan_stat542.csv", stringsAsFactors=FALSE)
testID <- read.csv("Project4_test_id.csv", stringsAsFactors=FALSE)

#######1.Define functions#######
#1) grap words with high frequency from factor with too many levels
ProcessLargeFactor <- function(featurelist, renumvec){
  myvocab = list()
  for(i in 1:length(featurelist)){
    prep_fun = tolower
    tok_fun = word_tokenizer
    it_data = itoken(featurelist[[i]],
                     preprocessor = prep_fun, 
                     tokenizer = tok_fun) #var1: adjust feature to be converted
    stop_words = c("i", "me", "my", "myself", 
                   "we", "our", "ours", "ourselves", 
                   "you", "your", "yours", 
                   "their", "they", "his", "her", 
                   "she", "he", "a", "an", "and",
                   "is", "was", "are", "were", 
                   "him", "himself", "has", "have", 
                   "it", "its", "of", "one", "for", 
                   "the", "us", "this")
    vocab = create_vocabulary(it_data, stopwords = stop_words)
    pruned_vocab = prune_vocabulary(vocab,
                                    term_count_min = 5, 
                                    doc_proportion_max = 0.5,
                                    doc_proportion_min = 0.001)
    bigram_vectorizer = vocab_vectorizer(pruned_vocab)
    dtm_data = create_dtm(it_data, bigram_vectorizer)
    v.size = dim(dtm_data)[2]
    ydata = as.numeric(data$loan_status)
    
    summ = matrix(0, nrow=v.size, ncol=4)
    summ[,1] = colapply_simple_triplet_matrix(
      as.simple_triplet_matrix(dtm_data[ydata==1, ]), mean)
    summ[,2] = colapply_simple_triplet_matrix(
      as.simple_triplet_matrix(dtm_data[ydata==1, ]), var)
    summ[,3] = colapply_simple_triplet_matrix(
      as.simple_triplet_matrix(dtm_data[ydata==0, ]), mean)
    summ[,4] = colapply_simple_triplet_matrix(
      as.simple_triplet_matrix(dtm_data[ydata==0, ]), var)
    
    n1=sum(ydata); 
    n=length(ydata)
    n0= n - n1
    
    myp = (summ[,1] - summ[,3])/
      sqrt(summ[,2]/n1 + summ[,4]/n0)
    
    words = colnames(dtm_data)
    id = order(abs(myp), decreasing=TRUE)[1:renumvec[i]] #var2: adjust top levels
    myvocab[[i]] = words[id]
  }
  names(myvocab) = names(featurelist)
  return(myvocab)
}

#2) log-loss function
logLoss = function(y, p){
  if (length(p) != length(y)){
    stop('Lengths of prediction and labels do not match.')
  }
  
  if (any(p < 0)){
    stop('Negative probability provided.')
  }
  
  p = pmax(pmin(p, 1 - 10^(-15)), 10^(-15))
  mean(ifelse(y == 1, -log(p), -log(1 - p)))
}


###################################################
#######2.Data Preprocess##########
# Step1: set loan_status
data$loan_status[which(data$loan_status %in% c('Default', 'Charged Off'))] = 1
data$loan_status[which(data$loan_status == 'Fully Paid')] = 0

# Setp2: deal with missing data
q <- numeric(30)
for (i in 1:30){
  q[i] <- sum(is.na(data[, i]))
}
data.na <- which(q > 0)

for (j in 1:3){
  data[is.na(data[, data.na[j]]), data.na[j]] <- "Unkown"
}
for (j in 4:7){
  data[is.na(data[, data.na[j]]), data.na[j]] <- median(data[,data.na[j]], na.rm = T)
}

# Step3: deal with factor levels
# 3.1 convert: emp_title (205137), title (43275) -- get top words
largefactor = list(data$emp_title, data$title)
names(largefactor) = c("emp_title", "title")
renum = c(50,50)
myvocab = ProcessLargeFactor(largefactor, renum)

# 3.2 drop: zip_code (919) and id
data = data[ ,-which(colnames(data) %in% c("zip_code"))]

# 3.3 seperate: earliest_cr_line (708)
earliest = as.data.frame(matrix(unlist(strsplit(data$earliest_cr_line, '-')), dim(data)[1], 2, byrow = T))
colnames(earliest) = c("earliest_cr_line_month", "earliest_cr_line_year")
data = data[ , -which(colnames(data) == "earliest_cr_line")]
data = cbind(data, earliest)

# Step4: obtain train and test data set
xgb.logloss <- numeric(3)
runtime = list()
for(s in 1:3){
  test <- data[which(data$id %in% testID[, s]), ]
  train <- data[-which(data$id %in% testID[, s]), ]
  
  # 4.1 create dtm_train dtm_test
  largefactors = list(train$emp_title, test$emp_title, train$title, test$title)
  dtm_train = list()
  dtm_test = list()
  for(i in 1:2){
    it_train = itoken(largefactors[[i*2-1]],
                      preprocessor = tolower, 
                      tokenizer = word_tokenizer)
    it_test = itoken(largefactors[[i*2]],
                     preprocessor = tolower, 
                     tokenizer = word_tokenizer)
    
    vocab = create_vocabulary(it_train)
    vocab = vocab[vocab$term %in% myvocab[[i]], ]
    bigram_vectorizer = vocab_vectorizer(vocab)
    dtm_train[[i]] = create_dtm(it_train, bigram_vectorizer)
    dtm_test[[i]] = create_dtm(it_test, bigram_vectorizer)
  }
  
  newtrain = train[, -which(colnames(train) %in% c("emp_title", "title"))]
  newtest = test[, -which(colnames(test) %in% c("emp_title", "title"))]
  numcoltrain = dim(newtrain)[2]
  numcoltest = dim(newtest)[2]
  
  newtrain = cbind(newtrain, as.matrix(dtm_train[[1]]), as.matrix(dtm_train[[2]]))
  newtest = cbind(newtest, as.matrix(dtm_test[[1]]), as.matrix(dtm_test[[2]]))
  
  # 4.2 obtain x and y
  ycol = which(colnames(newtrain) == "loan_status")
  train.x = newtrain[, -ycol]
  train.x = train.x[,-which(colnames(train.x) == "id")]
  train.x <- mutate_if(train.x, is.character, as.factor)
  #train.x = cbind(train.x[1:(numcoltrain-1)], apply(train.x[numcoltrain:dim(train.x)[2]],2,as.factor))
  train.y = as.numeric(newtrain[, ycol])
  
  ycol = which(colnames(newtest) == "loan_status")
  test.x = newtest[, -ycol]
  test.x = test.x[,-which(colnames(test.x) == "id")]
  test.x <- mutate_if(test.x, is.character, as.factor)
  #test.x = cbind(test.x[1:(numcoltest-1)], apply(test.x[numcoltest:dim(test.x)[2]],2,as.factor))
  test.y = as.numeric(newtest[, ycol])
  
  dtrain <- xgb.DMatrix(data = data.matrix(train.x),label = train.y)
  dtest <- xgb.DMatrix(data = data.matrix(test.x),label = test.y)
  
  start <- Sys.time()
  xgbmodel <- xgboost(data=dtrain, max.depth=3, eta=0.2, nround=200, verbose = FALSE,
                      objective="binary:logistic")
  phat <- predict(xgbmodel,data.matrix(test.x))
  end <- Sys.time()
  write.table(cbind(newtest$id, phat), paste("mysubmission_test",s,".txt", sep = ""), append = FALSE, sep = ", ", dec = ".",
              row.names = FALSE, col.names = c("id","prob"))
  xgb.logloss[s] <- logLoss(test.y, phat)
  runtime[[s]] <- end - start
}


###################################################
#######3.Predict Q3,Q4##########
# Step 1. obtain data from Q3, Q4
data3 <- read.csv("LoanStats_2018Q3.csv", stringsAsFactors=FALSE)
data4 <- read.csv("LoanStats_2018Q4.csv", stringsAsFactors=FALSE)
select.var <- c("id","loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_title",
                "emp_length", "home_ownership", "annual_inc", "verification_status","loan_status","purpose", "title",
                "zip_code", "addr_state", "dti", "earliest_cr_line", "fico_range_low", "fico_range_high",
                "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status", 
                "application_type", "mort_acc", "pub_rec_bankruptcies")
data3 <- data3[, select.var]
data4 <- data4[, select.var]

# Step 2. Process missing value
m <- numeric(length(select.var))
for (i in 1:length(select.var)){
  m[i] <- sum(is.na(data3[, i]))
}
data3.na <- which(m > 0)

data3[is.na(data3[, data3.na[1]]), data3.na[1]] <- "Unkown"
data3[is.na(data3[, data3.na[2]]), data3.na[2]] <- median(data3[,data3.na[2]], na.rm = T)

n <- numeric(length(select.var))
for (i in 1:length(select.var)){
  n[i] <- sum(is.na(data4[, i]))
}
data4.na <- which(n > 0)

data4[is.na(data4[, data4.na[1]]), data4.na[1]] <- median(data4[,data4.na[1]], na.rm = T)

# Step 3. Filter of response, set loan_status, convert column type
id3 <- which(data3$loan_status %in% c('Default', 'Charged Off', 'Fully Paid'))
data3 <- data3[id3,]
data3$loan_status[which(data3$loan_status %in% c('Default', 'Charged Off'))] = 1
data3$loan_status[which(data3$loan_status == 'Fully Paid')] = 0


id4 <- which(data4$loan_status %in% c('Default', 'Charged Off', 'Fully Paid'))
data4 <- data4[id4,]
data4$loan_status[which(data4$loan_status %in% c('Default', 'Charged Off'))] = 1
data4$loan_status[which(data4$loan_status == 'Fully Paid')] = 0

# Step 4. Deal with factor levels
# 3.1 drop: zip_code and id
data3 = data3[ ,-which(colnames(data3) %in% c("zip_code"))]
data4 = data4[ ,-which(colnames(data4) %in% c("zip_code"))]

# 3.2 seperate: earliest_cr_line 
earliest = as.data.frame(matrix(unlist(strsplit(data3$earliest_cr_line, '-')), dim(data3)[1], 2, byrow = T))
colnames(earliest) = c("earliest_cr_line_month", "earliest_cr_line_year")
data3 = data3[ , -which(colnames(data3) == "earliest_cr_line")]
data3 = cbind(data3, earliest)

earliest = as.data.frame(matrix(unlist(strsplit(data4$earliest_cr_line, '-')), dim(data4)[1], 2, byrow = T))
colnames(earliest) = c("earliest_cr_line_month", "earliest_cr_line_year")
data4 = data4[ , -which(colnames(data4) == "earliest_cr_line")]
data4 = cbind(data4, earliest)

# Step 5. get test Q3 Q4
# 5.1 create dtm_train dtm_test
largefactors = list(data3$emp_title, data4$emp_title, data3$title, data4$title)
dtm_data3 = list()
dtm_data4 = list()
for(i in 1:2){
  it_data3 = itoken(largefactors[[i*2-1]],
                    preprocessor = tolower, 
                    tokenizer = word_tokenizer)
  it_data4 = itoken(largefactors[[i*2]],
                    preprocessor = tolower, 
                    tokenizer = word_tokenizer)
  
  vocab = create_vocabulary(it_data3)
  vocab = vocab[vocab$term %in% myvocab[[i]], ]
  bigram_vectorizer = vocab_vectorizer(vocab)
  dtm_data3[[i]] = create_dtm(it_data3, bigram_vectorizer)
  vocab = create_vocabulary(it_data4)
  vocab = vocab[vocab$term %in% myvocab[[i]], ]
  bigram_vectorizer = vocab_vectorizer(vocab)
  dtm_data4[[i]] = create_dtm(it_data4, bigram_vectorizer)
}

newdata3 = data3[, -which(colnames(data3) %in% c("emp_title", "title"))]
newdata4 = data4[, -which(colnames(data4) %in% c("emp_title", "title"))]
numcoltrain = dim(newdata3)[2]
numcoltest = dim(newdata4)[2]

newdata3 = cbind(newdata3, as.matrix(dtm_data3[[1]]), as.matrix(dtm_data3[[2]]))
newdata4 = cbind(newdata4, as.matrix(dtm_data4[[1]]), as.matrix(dtm_data4[[2]]))


# 4.2 obtain x and y
ycol = which(colnames(newdata3) == "loan_status")
data3.x = newdata3[, -ycol]
data3.x = data3.x[ , -which(colnames(data3.x) == "id")]
data3.x <- mutate_if(data3.x, is.character, as.factor)
data3.x$int_rate <- as.numeric(data3.x$int_rate)
data3.x$revol_util <- as.numeric(data3.x$revol_util)
data3.y = as.numeric(newdata3[, ycol])

ycol = which(colnames(newdata4) == "loan_status")
data4.x = newdata4[, -ycol]
data4.x = data4.x[ , -which(colnames(data4.x) == "id")]
data4.x <- mutate_if(data4.x, is.character, as.factor)
data4.x$int_rate <- as.numeric(data4.x$int_rate)
data4.x$revol_util <- as.numeric(data4.x$revol_util)
data4.y = as.numeric(newdata4[, ycol])

# Step 5. get original 2017-2018Q2 data
data0.x <- rbind(train.x, test.x)
data0.y <- append(train.y, test.y)
dtrain0 <- xgb.DMatrix(data = data.matrix(data0.x),label = data0.y)
classifer <- xgboost(data=dtrain0, max.depth=3, eta=0.2, nround=200, verbose = FALSE,
                     objective="binary:logistic")

a=matrix(0,dim(data3.x)[1],dim(train.x)[2])
colnames(a) <- colnames(data0.x)
for (i in 1:dim(data3.x)[2]){
  a[,colnames(data3.x)[i]]=data3.x[,i]
}
data3.x = a

b=matrix(0,dim(data4.x)[1],dim(train.x)[2])
colnames(b) <- colnames(data0.x)
for (i in 1:dim(data4.x)[2]){
  b[,colnames(data4.x)[i]]=data4.x[,i]
}
data4.x = b

# Step 6. Prediction
phat3 <- predict(classifer, data.matrix(data3.x))
write.table(cbind(newdata3$id, phat3), paste("mysubmission_2018Q3.txt", sep = ""), append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("id","prob"))
xgb.logloss3 <- logLoss(data3.y, phat3)

phat4 <- predict(classifer, data.matrix(data4.x))
write.table(cbind(newdata4$id, phat4), paste("mysubmission_2018Q4.txt", sep = ""), append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("id","prob"))
xgb.logloss4 <- logLoss(data4.y, phat4)