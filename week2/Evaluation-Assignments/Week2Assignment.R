## Create data set to predict Emergency Department Admissions to hospital in an 8 hour shift

## libraries
library(h2o)

## Create Response variable with mean of 20 and sd of 6 
set.seed(528)
EDAdm <- round(rnorm(1000, mean  = 20, sd = 6),0)

## features used to predict ED admissions:

AvgAge <- round((EDAdm *6) + runif(1,1,4), 0) # Average age of patients seen
PtsSeen <- round((3.5*EDAdm) + runif(1,0,6), 0) # Number of patients seen in 8 hr shift
SOB <- round(.3*PtsSeen, 0) # of patients with shortness of breath in 8 hr shift
CPain <- round(.2*PtsSeen, 0) # of patients seen with chest pain in 8 hr shift
URI <- round(runif(1000, 8, 15), 3) # of patients with cold symptoms in 8 hr shift- not correlated with admissions

##   build the dataframe 
data <- data.frame(cbind(EDAdm, AvgAge, PtsSeen, SOB, CPain, URI))
y <- "EDAdm"
x <- setdiff(names(data), c("id", y))


## log onto h20 unpload dataframe

h2o.init()
data.hex <- as.h2o(data, destination_frame = "data.hex")


##split into training and test subsets.  Will perform xval

split <- h2o.splitFrame(data.hex, destination_frames = c("train", "test"))
train <- split[[1]]
test <- split[[2]]
y <- "EDAdm"
x <- setdiff(names(train), c("id", y))

## build and assess a simple model

mdl1 <- h2o.gbm(x,y, train, model_id = "model1", nfolds = 10, ntrees = 50, max_depth = 3)
h2o.performance(mdl1, train = T)
h2o.performance(mdl1, xval = T)
h2o.performance(mdl1, test)


## build and assess a complex, overfit  model

mdl2 <- h2o.gbm(x,y, train, model_id = "model1", nfolds = 10, ntrees = 700, max_depth = 6)
h2o.performance(mdl2, train = T)
h2o.performance(mdl2, xval = T)
h2o.performance(mdl2, test)


