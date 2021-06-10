################################################################################################
### Step1. Preprocessing
### Feature Selection을 진행하기 이전
### data의 overfitting 및 training time 감소를 위한 전처리를 진행하도록 함
################################################################################################
# Order
# (1) Standardization
# (2) Outlier detection and remove(with z-score)
# (3) Normalization
################################################################################################

### (1) standardization : zero-centered data

HV.scl <- as.data.frame(scale(HV), class = HV[, 101])
HV.log.scl <- as.data.frame(scale(HV.log),  class = HV[,101])

par(mfrow=c(1,2))
hist(HV.log.scl$X89)
hist(HV.log.scl$X14)


### (2) Outlier detection and remove(with z-score)

mydata.new <- HV

for(i in 1:100){
  #detect outliers
  mydata <- HV[,i]
  outlier_values <- boxplot.stats(mydata)$out
  
  
  #changing outliers to NA
  idx <- which(mydata %in% outlier_values)
  #cat(idx,",",i,"\n")
  #mydata.new <- as.data.frame(apply(mydata.new, 2, function(idx){idx[idx %in% boxplot(idx, plot = FALSE)$out] = NA; idx}))
}

dim(mydata.new)
str(idx)

HV.rm <- HV[-idx,]
dim(HV.rm)




### (3) normalization : 0~1 scaled data

normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}



HV.norm <- data.frame(normalize(HV[,-101]),  class = HV$class)
HV.log.norm <- data.frame(normalize(HV.log[,-101]),  class = HV$class)


HV.rm.norm <- data.frame(normalize(HV.rm[,-101]), class = HV.rm$class)
log.HV.rm.norm <- data.frame(normalize(log(HV.rm[,-101])), class = HV.rm$class)


#HV.na.norm <- data.frame(normalize(HV.rm.na[,-101]), class = HV.rm.na$class)
#log.HV.na.norm <- data.frame(normalize(log(HV.rm.na[,-101])), class = HV.rm.na[,101])


