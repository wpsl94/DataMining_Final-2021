#result model


#1. raw dataset
HV
lm1 <- lm(class~., data=HV)



#2. preprocessed removed dataset
log.HV.rm.norm
lm2 <- lm(class~., data=log.HV.rm.norm)


#3. Custom HV test model 
HV.model
HV.model.raw
lm3 <- lm(class~., data=HV.model.raw)


#4. hill climbing search
HV.hcs <- HV[,c(1, 2, 3, 7,9,10,12,13,14,17,18,19,20,21,23,24,25,26,27,28,29,30,35,36,38,40,41,42,43,44,46,47,49,50,52,54,57,58,62,64,65,71,73,81,82,83,84,86,87,88,91,94,96,97,98,99, 101)]
dim(HV.hcs)
lm4 <- lm(class~., data=HV.hcs)


#5. best first search
HV.bfs <- HV[,c(3,6,9,16,90,101)]
dim(HV.bfs)
lm5 <- lm(class~., data=HV.bfs)




#6. backward search
HV.bs <- HV[, c(1:74, 76:101)]
dim(HV.bs)
lm6 <- lm(class~., data=HV.bs)




#7.forward search
HV.fs <- HV[, c(3,6,9,16,90,101)]
dim(HV.fs)
lm7 <- lm(class~., data=HV.fs)





#multi linear
summary(lm1)
summary(lm2)
summary(lm3)
summary(lm4)
summary(lm5)
summary(lm6)
summary(lm7)

--------------------------------------------

#accuracy
accuracy(lm1)
accuracy(lm2)
accuracy(lm3)
accuracy(lm4)
accuracy(lm5)
accuracy(lm6)
accuracy(lm7)


glm()
