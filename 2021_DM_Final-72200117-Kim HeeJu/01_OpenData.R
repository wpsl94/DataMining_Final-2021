## Ready for Data "Hill Valley"

HV <- read.csv("Hill_Valley.csv")

str(HV)
class(HV)


is.na(HV)
sum(is.na(HV)) ##NA is not here
summary(HV)

#class 의 종류는 0과 1로 나뉘어져 있음


##중복데이터 제거

ds <- HV[,-101]
dup = which(duplicated(ds))
dup

#중복데이터 없음


## 데이터셋의 Skewness 수준을 확인

HV.skew <- c(1:100)

for(i in 1:100){

  HV.skew[i] <- skewness(HV[,i])
  
  cat("X", i, "skewness: ", HV.skew[i], "\n")
  
}

max(HV.skew)
par(mfrow=c(1,2))
hist(HV$X89)
hist(HV$X14)



#log transform
HV.temp <- HV[,-101]
HV.log <- log(HV.temp)

HV.log

HV.log <- data.frame(HV.log, class = HV[,101])


##데이터셋의 범위의 편차가 너무 크고 편향되어 있어 
##Scaling 및 Normalization이 필요함


