if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  "text2vec",
  "glmnet",
  "pROC",
  "slam",
  "xgboost",
  "MASS"
)
set.seed(500)

myvocab = scan(file = "myvocab.txt", what = character())

all = read.table("Project2_data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("Project2_splits.csv", header = T)

train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
vocab = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)

set.seed(500)
##########
#1. lasso#
##########
cv.out <- cv.glmnet(dtm_train, train[, 2], alpha = 1, family = 'binomial')
lasso.pred <-predict(cv.out, newx = dtm_test, lambda = cv.out$lambda.1se, type = "response")
write.table(cbind(test$new_id, lasso.pred), "mysubmission1.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("new_id","prob"))

set.seed(500)
###################
#2. xgboost and NB#
###################
train.y = train$sentiment
test.y = test$sentiment
param = list(max_depth = 2, 
             subsample = 0.5, 
             objective='binary:logistic')
ntrees = 1800
bst = xgb.train(params = param, 
                data = xgb.DMatrix(data = dtm_train, label = train.y),
                nrounds = ntrees, 
                nthread = 2)

dt = xgb.model.dt.tree(model = bst)
words = unique(dt$Feature[dt$Feature != "Leaf"])

new_feature_train = xgb.create.features(model = bst, dtm_train)
new_feature_train = new_feature_train[, - c(1:ncol(dtm_train))]
new_feature_test = xgb.create.features(model = bst, dtm_test)
new_feature_test = new_feature_test[, - c(1:ncol(dtm_test))]

trainY = as.factor(train[,2])
p=new_feature_train@Dim[2]; # p: number of predictors
y.levels = levels(trainY)
K= length(y.levels) # number of groups
mymean = matrix(0, K, p)  
mysd = matrix(0, K, p)
for(k in 1:K){
  mymean[k,] = apply(new_feature_train[trainY == y.levels[k],], 2, mean)
  mysd[k,] = apply(new_feature_train[trainY == y.levels[k],], 2, sd)
}
w=mean(trainY==y.levels[1])

#use NB to do prediction
ntest = dim(new_feature_test)[1]
newtmp1 = rep(0, ntest); newtmp2=rep(0, ntest)
testX = new_feature_test
for(i in 1:p){
  if(mysd[1,i] > 0){
    newtmp1 = newtmp1 - log(mysd[1,i]) - (testX[,i] - mymean[1,i])^2/(2*mysd[1,i]^2)
  }
  if(mysd[2,i] > 0){
    newtmp2 = newtmp2 - log(mysd[2,i]) - (testX[,i] - mymean[2,i])^2/(2*mysd[2,i]^2)
  }
}
newtmp1 = newtmp1 + log(w)
newtmp2 = newtmp2 + log(1-w)
diff = newtmp2-newtmp1
nb.pred = 1/(1 + exp(diff))
write.table(cbind(test$new_id, nb.pred), "mysubmission2.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("new_id","prob"))

set.seed(500)
########
#3. LDA#
########
lda.model = lda(dtm_train, train[, 2])
lda.pred = predict(lda.model, dtm_test)$posterior[ ,2]
write.table(cbind(test$new_id, lda.pred), "mysubmission3.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("new_id","prob"))

