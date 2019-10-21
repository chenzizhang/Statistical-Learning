if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  "glmnet",
  "xgboost",
  "mgcv",
  "parallel"
)

train.dat <- read.csv("train.csv")
test.dat <- read.csv("test.csv")
set.seed(100)
#########################
### define functions ###
#########################

ProcessMissingFixVal = function(train.col, test.col, replaceval){
  train.col[which(is.na(train.col))] = replaceval
  test.col[which(is.na(test.col))] = replaceval
  return(list(train = train.col, test = test.col))
}

ProcessRemoveVars = function(input, dropvar){
  return(input[, - which(names(input) %in% dropvar)])
}

ProcessWinsorization = function(train.data, test.data, quant){
  train.lim = quantile(train.data, quant)
  
  train.data[which(train.data > train.lim)] = train.lim
  test.data[which(test.data > train.lim)] = train.lim
  
  return(list(train = train.data, test = test.data))
}

PreProcessingMatrixOutput <- function(train.data, test.data){
  # generate numerical matrix of the train/test
  # assume train.data, test.data have the same columns
  categorical.vars <- c(colnames(train.data)[which(sapply(train.data, 
                                                          function(x) is.factor(x)))])
  train.matrix <- train.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  test.matrix <- test.data[, !colnames(train.data) %in% categorical.vars, drop=FALSE]
  n.train <- nrow(train.data)
  n.test <- nrow(test.data)
  for(var in categorical.vars){
    mylevels <- levels(train.data[, var])
    m <- length(mylevels)
    tmp.train <- matrix(0, n.train, m)
    tmp.test <- matrix(0, n.test, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.train[train.data[, var]==mylevels[j], j] <- 1
      tmp.test[test.data[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.train) <- col.names
    colnames(tmp.test) <- col.names
    train.matrix <- cbind(train.matrix, tmp.train)
    test.matrix <- cbind(test.matrix, tmp.test)
  }
  return(list(train = as.matrix(train.matrix), test = as.matrix(test.matrix)))
}


##############
#XGBoost
##############

#1. remove var
remove.var <- c('PID', 'Sale_Price')
train.y <- log(train.dat$Sale_Price)
train.x <- ProcessRemoveVars(train.dat, remove.var)
test.PID <- test.dat$PID
test.x <- ProcessRemoveVars(test.dat, remove.var)
  
#2. process missing val
r <- ProcessMissingFixVal(train.dat$Garage_Yr_Blt, test.dat$Garage_Yr_Blt, 0)
train.x$Garage_Yr_Blt <- r$train
test.x$Garage_Yr_Blt <- r$test
  
#3. process binary model
r = PreProcessingMatrixOutput(train.x, test.x)
train.x = r$train
test.x = r$test
  
dtrain <- xgb.DMatrix(data = train.x,label = train.y) 
dtest <- xgb.DMatrix(data = test.x)
  
xgbmodel <- xgboost(data = dtrain, nrounds = 300, verbose = FALSE, max_depth = 3, eta = 0.2)
xgbpred <- predict (xgbmodel,dtest)

write.table(cbind(test.PID, exp(xgbpred)), "mysubmission1.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("PID","Sale_Price"))

#########################
### Lasso ###
#########################
#1. remove var
remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating',
                  'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                  'Longitude','Latitude',"PID", "Sale_Price")
train.y <- log(train.dat$Sale_Price)
train.x <- ProcessRemoveVars(train.dat, remove.var)
test.PID <- test.dat$PID
test.x <- ProcessRemoveVars(test.dat, remove.var)

#2. set "Mo_Sold" and "Year_Sold" as categorical variables
train.x$Mo_Sold <- as.factor(train.x$Mo_Sold)
test.x$Mo_Sold <- as.factor(test.x$Mo_Sold)
train.x$Year_Sold <- as.factor(train.x$Year_Sold)
test.x$Year_Sold <- as.factor(test.x$Year_Sold)
  
#3. process missing val
r <- ProcessMissingFixVal(train.dat$Garage_Yr_Blt, test.dat$Garage_Yr_Blt, 0)
train.x$Garage_Yr_Blt <- r$train
test.x$Garage_Yr_Blt <- r$test
  
#4. process catergorical var
r <- PreProcessingMatrixOutput(train.x, test.x)
train.x <- r$train
test.x <- r$test
  
#5. process winsor var
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", 
                   "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', 
                   "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", 
                   "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
for(var in winsor.vars){
  r <- ProcessWinsorization(train.x[, var], test.x[, var], 0.95)
  train.x[, var] <- r$train
  test.x[, var] <- r$test
}
  
#6. construct lasso.model and predict
cv.out <- cv.glmnet(train.x, train.y, alpha = 1)
lambdamin <- cv.out$lambda.min
lasso.model <- glmnet(train.x,train.y,lambda = lambdamin, alpha = 1)
coef.matrix <- as.matrix(lasso.model$beta)
coef.idx <- which(coef.matrix != 0)
lasso.beta <- rownames(coef.matrix)[coef.idx]
lasso.pred <-predict(lasso.model, newx = test.x)

write.table(cbind(test.PID, exp(lasso.pred)), "mysubmission2.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("PID", "Sale_Price"))


#########################
### GAM ###
#########################
#1. remove var
remove.var <- c('Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating',
                  'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                  'Longitude','Latitude', 'Mo_Sold', 'Year_Sold',
                  'PID', 'Sale_Price')
train.y <- log(train.dat$Sale_Price)
train.x <- ProcessRemoveVars(train.dat, remove.var)
test.PID <- test.dat$PID
test.x <- ProcessRemoveVars(test.dat, remove.var)
  
#2. process missing val
r <- ProcessMissingFixVal(train.dat$Garage_Yr_Blt, test.dat$Garage_Yr_Blt, 0)
train.x$Garage_Yr_Blt <- r$train
test.x$Garage_Yr_Blt <- r$test
  
#3. process linear val
categorical.vars <- colnames(train.x)[which(sapply(train.x, 
                                                     function(x) is.factor(x)))]
num.vars <- names(train.x)
num.vars <- num.vars[num.vars != "Sale_Price"]
num.vars <- num.vars[! num.vars %in% categorical.vars]
linear.vars <- c('BsmtFin_SF_1', 'Bsmt_Full_Bath', 
                 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 
                 'Kitchen_AbvGr', 'Fireplaces', 'Garage_Cars')
num.vars <- num.vars[! num.vars %in% linear.vars]
  
#4. process level val
select.level.var <- c("MS_SubClass__Duplex_All_Styles_and_Ages", "MS_SubClass__One_Story_1945_and_Older","MS_SubClass__One_Story_1946_and_Newer_All_Styles","MS_SubClass__Split_Foyer",
                      "MS_SubClass__Two_and_Half_Story_All_Ages", "MS_SubClass__Two_Story_1945_and_Older", 
                      "MS_SubClass__Two_Story_PUD_1946_and_Newer", 
                      "MS_Zoning__C_all", "MS_Zoning__I_all", "MS_Zoning__Residential_Medium_Density",   
                      "Land_Contour__Bnk", "Land_Contour__HLS", "Land_Contour__Low",
                      "Lot_Config__Corner",
                      "Lot_Config__CulDSac", "Lot_Config__FR2",  "Lot_Config__Inside",                       
                      "Land_Slope__Gtl", "Land_Slope__Sev",                          
                      "Neighborhood__Brookside", "Neighborhood__Clear_Creek",                
                      "Neighborhood__Crawford", "Neighborhood__Edwards",                    
                      "Neighborhood__Green_Hills", "Neighborhood__Greens",                     
                      "Neighborhood__Meadow_Village", "Neighborhood__Mitchell",                   
                      "Neighborhood__North_Ames", "Neighborhood__Northridge",                 
                      "Neighborhood__Northridge_Heights", "Neighborhood__Old_Town",                   
                      "Neighborhood__Somerset",                   
                      "Neighborhood__Stone_Brook", "Neighborhood__Timberland",                 
                      "Condition_1__Feedr", "Condition_1__Norm","Condition_1__PosA", "Condition_1__RRAe",                        
                      "Bldg_Type__Duplex", "Bldg_Type__OneFam", "Bldg_Type__Twnhs", 
                      "House_Style__One_and_Half_Fin","House_Style__Two_and_Half_Fin",
                      "Overall_Qual__Average", "Overall_Qual__Below_Average",              
                      "Overall_Qual__Excellent", "Overall_Qual__Fair",                       
                      "Overall_Qual__Good", "Overall_Qual__Poor",                       
                      "Overall_Qual__Very_Excellent", "Overall_Qual__Very_Good",                  
                      "Overall_Qual__Very_Poor", 
                      "Overall_Cond__Average", "Overall_Cond__Below_Average", 
                      "Overall_Cond__Excellent", "Overall_Cond__Fair", 
                      "Overall_Cond__Good", "Overall_Cond__Poor", 
                      "Overall_Cond__Very_Good",                  
                      "Exterior_1st__AsbShng", "Exterior_1st__BrkFace",                    
                      "Mas_Vnr_Type__CBlock", "Mas_Vnr_Type__Stone",
                      "Foundation__BrkTil", "Foundation__PConc", "Foundation__Stone",                       
                      "Bsmt_Qual__Excellent","Bsmt_Qual__Fair", 
                      "Bsmt_Cond__Fair", "Bsmt_Cond__Good",                          
                      "Bsmt_Exposure__Av", "Bsmt_Exposure__Gd", "Bsmt_Exposure__No",
                      "BsmtFin_Type_1__GLQ", "BsmtFin_Type_1__LwQ",            
                      "Heating_QC__Excellent", "Heating_QC__Fair", "Heating_QC__Poor", "Heating_QC__Typical",
                      "Kitchen_Qual__Excellent", "Kitchen_Qual__Fair", "Kitchen_Qual__Typical",
                      "Functional__Maj2","Functional__Mod", "Functional__Sal", "Functional__Sev", "Functional__Typ",                         
                      "Garage_Type__Attchd", "Garage_Type__CarPort", "Garage_Type__More_Than_Two_Types",
                      "Garage_Qual__Excellent", "Garage_Qual__Good", "Garage_Cond__Fair", "Garage_Cond__Typical",                      
                      "Sale_Type__COD", "Sale_Type__Con", "Sale_Type__ConLI", 
                      "Sale_Condition__Abnorml", "Sale_Condition__AdjLand", "Sale_Condition__Family","Sale_Condition__Partial")

m <- length(select.level.var)
tmp.train <- matrix(0, nrow(train.x), m)
tmp.test <- matrix(0, nrow(test.x), m)
colnames(tmp.train) <- select.level.var
colnames(tmp.test) <- select.level.var
select.var <- numeric(m)
for(i in 1:m){
  tmp <- unlist(strsplit(select.level.var[i], '__'))
  select.var[i] <- tmp[1]
  select.level <- tmp[2]
  tmp.train[train.dat[, select.var[i]]==select.level, i] <- 1
  tmp.test[test.dat[, select.var[i]]==select.level, i] <- 1
}
  
gam.formula <- paste0("Sale_Price ~ ", linear.vars[1])
#select.binary.vars <- select.var
for(var in c(linear.vars[-1], select.level.var))
  gam.formula <- paste0(gam.formula, " + ", var)
for(var in num.vars)
  gam.formula <- paste0(gam.formula, " + s(", var, ")")
gam.formula <- as.formula(gam.formula)
  
# construct train and test dataframe
train.con <- cbind(train.x,train.y)
colnames(train.con)[ncol(train.con)] <- "Sale_Price"
  
train.dat.gam <- as.data.frame(cbind(train.con[ ,c("Sale_Price", linear.vars, num.vars)], tmp.train))
test.dat.gam<- as.data.frame(cbind(test.x[ ,c(linear.vars, num.vars)], tmp.test))
  
# construct gam.model and predict
gam.model <- gam(gam.formula, data = train.dat.gam, method="REML")
gam.pred <- predict.gam(gam.model, newdata = test.dat.gam)


write.table(cbind(test.PID, exp(gam.pred)), "mysubmission3.txt", append = FALSE, sep = ", ", dec = ".",
            row.names = FALSE, col.names = c("PID","Sale_Price"))

