### Step4. FS selection
### 가장 정제된 데이터셋을 기반으로 기본적으로 4가지의 algorithm을 기준으로 FS를 진행하고자 함
### 1. hill climbing search
### 2. best firsth search
### 3. backward search
### 4. forward search
### 단, exhausted search 의 경우 시간 소요가 너무 오래 걸려 제외함


HV.test <- log.HV.rm.norm
HV.test$class <- as.factor(HV.test$class)

dim(HV.test)
df <- HV.test

ds <- df[,-101]
cl <- df[,101]


start <- Sys.time()

#start
print("1. hill.climbing.search")

subset1 <- hill.climbing.search(names(ds), evaluator)
subset1
result.subset1 <- evaluator(subset1) # accuracy of subset



print("2. best.first.search")

subset2 <- best.first.search(names(ds), evaluator)
subset2
result.subset2 <- evaluator(subset2) # accuracy of subset



print("3. backward.search")

subset3 <- backward.search(names(ds), evaluator)
subset3
result.subset3 <- evaluator(subset3) # accuracy of subset


print("4. forward.search")

subset4 <- forward.search(names(ds), evaluator)
subset4
result.subset4 <- evaluator(subset4) # accuracy of subset


end <- Sys.time()
#end

end-start
