################################################################################################
### Step3. Logistic Regression
### class가 0과 1로 범주화되어 있음을 확인
### 로지스틱 회귀 진행
################################################################################################

newdata <- HV.model


mod.graw <- glm(class ~ ., data = newdata)
mod.graw

summary(mod.graw)

test <- HV.model[, -1]

pred <- predict(mod.graw, test)
pred <- round(pred, 0)
pred


answer <- HV.model$class

#test 
pred == answer
acc <- mean(pred==answer)
acc

plot(HV.model$X1,dnorm(HV.model$X1, mean=0, sd=1))

################################################################################################

newdata <- log.HV.rm.norm[,c(1:74, 76:101)]

test <- log.HV.rm.norm[,c(1:74, 76:101)]

answer <- log.HV.rm.norm[,101]





newdata <- HV.log.norm
newdata <- HV.na.norm
newdata <- HV


newdata <- HV.rm
newdata <- HV.rm.norm
newdata <- log.HV.rm.norm

newdata <- HV.model


mod.graw <- glm(class ~ ., data = newdata)
mod.graw

summary(mod.graw)

test <- HV.log.norm[, 1:100]
test <- HV.na.norm[, 1:100]
test <- HV[, 1:100]
test <- newdata.5

test <- HV.model[, -1]


test <- HV.rm[,1:100]
test <- HV.rm.norm[,1:100]
test <- log.HV.rm.norm[,1:100]

pred <- predict(mod.graw, test)
pred <- round(pred, 0)
pred

answer <- HV.log.norm$class
answer <- HV.na.norm$class
answer <- HV$class

answer <- newdata.5$class

answer <- HV.rm$class
answer <- HV.rm.norm$class
answer <- log.HV.rm.norm$class

answer <- HV.model$class

#test 
pred == answer
acc <- mean(pred==answer)
acc


