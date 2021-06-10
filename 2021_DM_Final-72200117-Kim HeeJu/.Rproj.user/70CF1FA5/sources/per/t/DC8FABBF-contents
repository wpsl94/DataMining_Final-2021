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
evaluator(subset1) # accuracy of subset



print("2. best.first.search")

subset2 <- best.first.search(names(ds), evaluator)
subset2
evaluator(subset2) # accuracy of subset



print("3. backward.search")

subset3 <- backward.search(names(ds), evaluator)
subset3
evaluator(subset3) # accuracy of subset


print("4. forward.search")

subset4 <- forward.search(names(ds), evaluator)
subset4
evaluator(subset4) # accuracy of subset


end <- Sys.time()
#end

end-start



mean(result.subset1, result.subset2, result.subset3, result.subset4)