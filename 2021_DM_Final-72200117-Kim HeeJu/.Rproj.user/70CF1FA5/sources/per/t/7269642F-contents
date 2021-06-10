### Step5. mRMR


##raw
HV.ds <- HV
HV.ds$class <- as.factor(HV$class)
str(HV.ds$class)

ds <- HV.ds[,-101]
cl <- HV.ds[,101]
sel <- MRMR(ds, cl, 100) # selected feature
sel
sel.feature1 <- names(sel$selection)
result1 <- evaluator(sel.feature1)


##pre-processed(oulier removed + log transformed + normalization)
log.HV.rm.norm.ds <- log.HV.rm.norm
log.HV.rm.norm.ds$class <- as.factor(log.HV.rm.norm.ds$class)

ds <- log.HV.rm.norm.ds[,-101]
cl <- log.HV.rm.norm.ds[,101]
sel <- MRMR(ds, cl, 100) # selected feature
sel
sel.feature2 <- names(sel$selection)
result2 <- evaluator(sel.feature2)


##made
ds <- HV.model[,-1]
cl <- HV.model[,1]
sel <- MRMR(ds, cl, 51) # selected feature
sel
sel.feature3 <- names(sel$selection)
result3 <- evaluator(sel.feature3)

summary(lm(class~., HV.model))

##


0.527


HV.test2 <- HV
HV.test2$class <- as.factor(HV.test2$class)
























### 1. outlier 제거 + log변환 + normalization 
ds <- log.HV.rm.norm[,-101]
cl <- log.HV.rm.norm[,101]
sel <- MRMR(ds, cl, 10) # 3: #of selected feature
sel
sel.feature <- names(sel$selection)
acc <- evaluator(sel.feature)
acc



### 2. outlier 제거 + normalization
ds <- HV.rm.norm[,-101]
cl <- HV.rm.norm[,101]
sel <- MRMR(ds, cl, 10) # 3: #of selected feature
sel
sel.feature <- names(sel$selection)
acc <- evaluator(sel.feature)
acc


### 3. raw data + log변환 
ds <- HV.log.norm[,-101]
cl <- HV.log.norm[,101]
sel <- MRMR(ds, cl, 10) # 3: #of selected feature
sel
sel.feature <- names(sel$selection)
acc <- evaluator(sel.feature)
acc


### 4. raw data
ds <- HV.scl[,-101]
cl <- HV.scl[,101]
sel <- MRMR(ds, cl, 10) # 3: #of selected feature
sel
sel.feature <- names(sel$selection)
acc <- evaluator(sel.feature)
acc