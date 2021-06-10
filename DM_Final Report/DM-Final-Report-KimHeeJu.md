---

*Title: "Feature Selection Report"*    

*Author: "72200117 Kim Hee Ju"*  

*Due date: '2021 6 11 '*  

------




<br>  

#### **시작하기 전에..**  

><details><summary></summary>현재 R Markdown 내부 오류 중 Knitr 라이브러리와 관련된 이슈가 발생하였습니다.Chunk가 실행이 되는 도중 Object를 찾지 못하는 에러가 발생했고, 이로 인해 현재 제 다양한 라이브러리와 참조변수가 많이 짜여져 있는 제 코드를 MArkdown v2에서는 Chunk단위로 실행시키기가 어려워 Data의 구성과 함께 설명드리며 결과를 보여드리기 불가한 점 미리 양해부탁드립니다.  <br>
>해당 이슈를 해결하기 위해해 문서를 찾아보는 중에, Markdown v1이후 kntir 패키지를 더 이상 따로 지원을 하지 않아 발생하는 오류 중 하나라고 하였습니다.   
>만약 비슷한 이슈가 있는 학생이 있거나, 솔루션을 아신다면 조언 부탁드립니다.   
>본 이슈를 찾아보면서 참고했던 링크를 함께 첨부해드립니다.  </details>

 * <https://rmarkdown.rstudio.com/authoring_migrating_from_v1.html>    
 * <https://community.rstudio.com/t/knitting-issue-object-not-found/23448>    
 * <https://github.com/yihui/knitr/issues/445>  

<br>
<br> <br>
<br>  

### **[Assignment]**
**Hill_Vally 데이터셋에 대해 가장 높은 분류 정확도를 제공하는 Feature의 집합과 Accuracy를 제시하시오**     

> * Hill_vally 데이터셋의 마지막 컬럼이 class label 임
> * Accuracy 의 평가는 10-fold cross validation 으로 하며, seed 값은 100 으로 함
> * 전처리 후 작업 진행
<br>

<br>

<br>

### **1. Prepare for Data Analysis**  

<br>

*`Hill_Valley dataset`* 의 특징은 다음과 같았으며, raw dataset의 평가정도는 다음과 같았다.

<br>

* **`"Hill_Valley" dataset` 의 특징**

  1. 1212 obs. of 101 variables. Dataframe.
  2. row별 숫자의 연관성은 있어보이지만, Col 간의 숫자 연관성은 크게 없어보임
  3. Class column의 type이 `Factor`가 아닌 `Integer`임
  4. Class의 구성은 0과 1의 두 종류로 구성되어 있는 이진 분류 데이터셋임

<br>

* 중복데이터는 **포함되어있지 않음**을 확인

```R
##중복데이터 제거

ds <- HV[,-101]
dup = which(duplicated(ds))
dup

#result = 중복데이터 없음
```

<br>

* 현재 데이터셋의 `skewness` 수준을 확인하여 편향된 수준을 확인

```R
HV.skew <- c(1:100)

for(i in 1:100){
  HV.skew[i] <- skewness(HV[,i]) 
  cat("X", i, "skewness: ", HV.skew[i], "\n")  
}
```

> 1. 최저 Skewness : 2.913802 [X89]
> 2. 최고 Skewness : 3.160558 [X14]
> 3. 모두 한쪽으로 *치우쳐져* 있는 양상을 볼 수 있음
> 4. 또한 데이터셋의 범위의 편차가 너무 크고 편향되어 있어 **Scaling** 및 **Normalization** 이 필요함

![](https://user-images.githubusercontent.com/19165180/121409430-77fbbd80-c99c-11eb-9b8c-c3d010f4405e.png)

<br>

* model 평가 결과

| Adjusted R-squared | p-value   |
| :----------------- | :-------- |
| 0.1041             | 6.812e-12 |
<br>
> * **Adjusted R-squared**
>   : 원본 모델의 경우 Outlier가 존재하여 Class를 각 Variable들이 0.1041만큼 설명할 수 있음 (10%)
> * **p-value**
>   : 해당 모델의 경우 6.812e-12만큼의 의미가 있음 (p-value<0.05 인 경우 신뢰수준이 95% 이상)
> * '*'가 1개 이상 표기된 변수는 100개중 총 4개

<br> 
<br>
<br>
    

### **2. Data Pre-Processing**

<br>

기존의 Hill_Valley 데이터셋은 정제되어 있지 않고, 부여된 Feature의 개수가 많아져 Modeling 시 Raw data의 모든 Feature를 사용하는 것은 자원(Computing Power, Memory 등) 측면에서 매우 비효율적일 수 있게 된다. 이는 이후 모델의 성능을 저하시키는 요인이 될 수 있기 때문에 일부 필요한 Feature를 추출하여 가장 높은 성능을 낼 수 있는 모델을 형성할 수 있도록 Feature Selection을 진행하여야 한다.   

Feature Selection을 만들기 위해 Feature Subset을 형성하기 위한 전처리 과정은 다음과 같다.  

<br>  

  **[1] Standardization** 

  **[2] Outlier detection and elimination**

  **[3] Normalization**  

<br>  

이러한 전처리과정을 통해 모델의 성능을 전반적으로 향상시킬 수 있도록 한다. 

여기서 *모델의 성능*은 다음을 지칭한다.

- `Data`의 `overfitting` 감소  
- `Training time` 감소  
- `Accuracy` 향상  

<br>  

****

<br>  

#### **(0) Library**

이번 데이터 분석을 위해 사용된 라이브러리는 다음과 같다.    

<br>


```r
library("varSelRF") # Variable Selection using Random Forests
library("caret") # SVM and Training
library("rJava") 
library("FSelector") # Feature Selection
library("mlbench") # machine learning benchmark
library("ggplot2") # data visualization
library("e1071") # skewness function
library("MASS") # Functions and datasets to support Venables and Ripley
library("praznik") # MRMR
```

<br>

<br>



#### **(1) Standardization : zero-centered data**    

**`Standardization`**란, 입력된 각각의 x 를 전체 x 의 평균 (mean) 으로 빼주고,  x의 표준 편차로 나누어주면, 이를 통해 정규 분포를 그릴 수 있는데, 이때 표준화는 해당 정규 분포를 평균이 0이고 분산이 1인 표준 정규 분포로 바꿔 내는 것이다.   



이러한 작업을 진행하는 이유는 다음과 같다.  

> * 변수마다 각 변수의 단위와 퍼져있는 정도가 다름  
> * 이로 인한 직접적인 비교가 불가함  
> * 표준화를 통해 정보들을 통일하여 각 변수들 간에 비교가 가능해짐  



`Standardization`은 내장되어 있는 함수인 `Scale`함수를 통해 진행하였다.   

또한 skewness때문에 편향되어 있어, log변환을 진행한 후, standardization을 진행해보기도 하였다.  



```R
HV.scl <- as.data.frame(scale(HV), class = HV[, 101])
HV.log.scl <- as.data.frame(scale(HV.log),  class = HV[,101])
```

<p align="center">![](https://user-images.githubusercontent.com/19165180/121421288-fa8a7a00-c9a8-11eb-9f36-a30a2d1f9e88.png)</p>
<br>

***

<br>
<br>

#### **(2) Outlier detection and Elimination**    

표준화를 진행하여 데이터셋에서 얼마나 동떨어진 데이터들이 있는 지(=`Outlier`)를 살펴본 후, 이를 제거해주는 작업을 진행하였다.  대체로 평균에서 매우 먼 값이며, 표준편차보다는 큰 값을 의미한다. 통상적으로 신뢰구간 밖의 범위의 값들을 Outlier로 정의하고, 해당 인덱스의 값들은 제외하거나 평균, 혹은 중앙값으로 대체하여 model을 재구성한다. 

또한 이상치로 인해 <u>**나머지 값들은 0에 가까운 숫자로 왜곡되게 정규화될 가능성이 높기 때문에**</u> 이상치 제거 과정을 거친 후 정규화 작업을 진행하도록 결정하였다.

본 실험에서는 후에 탐색 시간을 줄이기 위해 데이터의 수를 줄이기 위하여 대체보다 ***탐색 후 제거***를 하는 방향으로 진행하였다. 

```r
mydata.new <- HV.log.scl

for(i in 1:100){
  #detect outliers
  mydata <- HV.log.scl[,i]
  outlier_values <- boxplot.stats(mydata)$out
  
  #changing outliers to NA
  idx <- which(mydata %in% outlier_values)  
}

dim(mydata.new)
str(idx)

HV.rm <- HV.log.scl[-idx,]
dim(HV.rm)
```

![](https://user-images.githubusercontent.com/19165180/121422490-40940d80-c9aa-11eb-8c12-0f8a22abfc7a.png)

<br>

* 추출한 Outlier 값의 Index는 <u>*총 215개임*</u>

|                | **전** | 후                                     |
| -------------- | ------ | -------------------------------------- |
| **objects**    | 1212   | <span style="color:red">**997**</span> |
| **variaables** | 101    | <span style="color:red">**101**</span> |

<br>
<br>

***

<br>
<br>

#### **(3) Normalization : 0~1 Scaled data**    

정규화를 통해 데이터를 0과 1사이의 범위로 `scaling`하여 데이터간 단위가 달라 객관적인 비교가 어려운 상태의 데이터셋을, 직관적으로 볼 수 있도록 변환하였다. 

`Min-Max Scaling` 방법을 사용하였으며, 이는 최대값과 최소값을 사용하여 원 데이터의 최소값을 0, 최대값을 1로 만드는 방법이다. 

```r
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}
```



다양한 `Feature Selection` 결과들을 비교하기 위하여 다음과 같은 경우를 정규화 시도한 후, `Feature Selection` 및 `model evaluation` 시도를 진행해보았다.  



> 1. Raw Data  
> 2. Log Transformed Raw Data  
> 3. Outlier eliminated Raw Data  
> 4. Outlier eliminated Raw Data with Log Transforming  



```R
#1
HV.norm <- data.frame(normalize(HV[,-101]),  class = HV$class)
#2
HV.log.norm <- data.frame(normalize(HV.log[,-101]),  class = HV$class)
#3
HV.rm.norm <- data.frame(normalize(HV.rm[,-101]), class = HV.rm$class)
#4
log.HV.rm.norm <- data.frame(normalize(log(HV.rm[,-101])), class = HV.rm$class)
```

<br>

***

<br>
<br><br>

### **3. k-fold Cross Validation **

<br>

<p align="center">![](https://user-images.githubusercontent.com/19165180/121427834-20674d00-c9b0-11eb-96c0-7dd112e48848.png)</p>

**[Definition]**

* **K개의 Fold를 만들어서 진행하는 교차 검증**  

* 총 데이터 갯수가 적은 데이터 셋에 대해 Accuracy를 향상시킬 수 있음  

* Training / Validaition / Test 세 집단으로 분류하는 것 보다 Training 과 Test set으로만 분류할 때 학습 데이터 셋이 더 많아짐  

* 데이터 수가 적은데 검증과 테스트에 데이터를 더 뺐기면 undeerfitting 등 성능이 미달되는 모델이 학습됨  


<br>

<br>

![](https://user-images.githubusercontent.com/19165180/121428419-bc915400-c9b0-11eb-8252-af39e535f0c0.png)

**[Process]**

1. Training set과 Test set 으로 나눔  

2. Training을 K개의 fold로 나눔 (본 실험에서는 10개로 나누었음)  

3. 한 개의 fold에 있는 데이터를 다시 K개로 쪼갠다음, K-1개는 Training Data, 마지막 한개는 Validation Data set으로 지정  

4. 모델을 생성하고 예측을 진행하고, 이에 대한 에러값을 추출  

5. 다음 fold에서는 Validation set을 바꿔서 지정하고, 이전 fold 에서 Validation 역할을 했던 Set은 다시 Training set으로 활용  

6. 이를 K번 반복함  


<br>

***

<br>

```r
###10-fold cross validation evaluate function

evaluator <- function(subset) {
    
  set.seed(100)  #seed 값은 100으로 고정하도록 함
    
  K<- 10 #10-fold cross validation
  folds <- createFolds(1:nrow(ds), K)
    
  acc <- c()
    
  for (i in 1:K) {      
    ts.idx <- folds[[i]]
    ds.tr <- ds[-ts.idx,subset]
    ds.ts <- ds[ts.idx,subset]
    cl.tr <- cl[-ts.idx]
    cl.ts <- cl[ts.idx]
    model <- svm(ds.tr,cl.tr)
    pred <- predict(model, ds.ts)
    acc[i] <- mean(pred==cl.ts)
  }
    
  cat(mean(acc),subset, "\n")
  return(mean(acc))
    
}
```

<br>

<br>

<br>

### **4. Feature Selection**

<br>

Feature Selection은 다음과 같은 이유 때문에 진행이 된다.  


1. 더 나은 성능의 모델을 만들기 위해  
  
2. 이해하기 쉬운 모델을 만들기 위해  

3. 더 빠르게 실행되는 모델을 만들기 위해  


이를 위한 Feature Selection 방법에는 Filtering, Wrapper, Embedded의 3가지가 있다.  


본 연구에서는 Wrapper Method를 기반으로 진행이 되었으며, 이를 위해 FSelector Algorithm과 mRMR 기법을 활용하였다.  


Wrapper는 예측 모델을 사용하여 피처들의 부분 집합을 만들어 계속 테스트 하여 최적화된 피처들의 집합 만드는 방법으로, 최적화된 모델을 만들기 위해 여러번 모델을 생성하고 성능을 테스트 해야 하기 때문에 많은 시간이 필요하다.   


<br><br>

#### **(1) FSelector Algorithm**    

<br>

가장 정제된 데이터셋인 `log.HV.rm.norm` 데이터셋을 기반으로 기본적으로 4가지의 algorithm을 기준으로 FS를 진행하고자 함

`log.HV.rm.norm` : Outlier Removed Hill_Valley Dataset with Normalization and Log Transforming

> - hill climbing search
> - best firsth search
> - backward search
> - forward search

단, exhausted search 의 경우 시간 소요가 너무 오래 걸려 제외함

```r
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

end-start #Time difference of 2.08249 hours
```




|                           |                     hill.climbing.search                     |              best.first.search               |      backward.search       |                forward.search                |
| ------------------------- | :----------------------------------------------------------: | :------------------------------------------: | :------------------------: | :------------------------------------------: |
| **Selected<br />Feature** | "X1"  "X2"  "X3"  "X7"  "X9"  "X10" "X12" "X13" "X14" "X17" "X18" "X19" "X20" "X21" "X23" "X24" "X25" "X26" "X27" "X28" "X29" "X30" "X35" "X36" "X38" "X40" "X41" "X42" "X43" "X44" "X46" "X47" "X49" "X50" "X52" "X54" "X57" "X58" "X62" "X64" "X65" "X71" "X73" "X81"  "X82" "X83" "X84" "X86" "X87" "X88" "X91" "X94" "X96" "X97" "X98" "X99" |      **"X3"  "X6"  "X9"  "X16" "X90"**       | X1 : X74, <br />X76 : X100 |      **"X3"  "X6"  "X9"  "X16" "X90"**       |
| **count**                 |                              56                              |                      5                       |             99             |                      5                       |
| **accuracy**              |                          0.5274848                           | <span style="color:red">**0.5395152**</span> |         0.5264848          | <span style="color:red">**0.5395152**</span> |

* **Best result Algorithm : best first search & forward  search** (acc = 0.5395152)

<br>

<br>
<br>  

#### **(2) mRMR** 

```r
##raw
HV.ds <- HV
HV.ds$class <- as.factor(HV$class)
str(HV.ds$class)

ds <- HV.ds[,-101]
cl <- HV.ds[,101]
sel <- MRMR(ds, cl, 100) # selected feature
sel
sel.feature <- names(sel$selection)
result1 <- Kfold.eval(ds,cl,sel.feature)


##pre-processed(oulier removed + log transformed + normalization)
log.HV.rm.norm.ds <- log.HV.rm.norm
log.HV.rm.norm.ds$class <- as.factor(log.HV.rm.norm.ds$class)

ds <- log.HV.rm.norm.ds[,-101]
cl <- log.HV.rm.norm.ds[,101]
sel <- MRMR(ds, cl, 100) # selected feature
sel
sel.feature <- names(sel$selection)
result2 <- Kfold.eval(ds,cl,sel.feature)


##made
ds <- HV.model[,-1]
cl <- HV.model[,1]
sel <- MRMR(ds, cl, 51) # selected feature
sel
sel.feature <- names(sel$selection)
result3 <- Kfold.eval(ds,cl,sel.feature)
```

<br>

<br>

|                           | raw data                                                     | preprocessed data                                            | *custom data                                                 |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Selected<br />Feature** | X4 X15 X73 X23 X14 X41 X16 X52 X24 X74 X13 X81 X61 X38 X75 X17 X42 X18 X86 X72 X59 X40 X69 X12 X80 X63 X36 X22 X78 X51 X37 X76 X25 X62 X46 X83 X20 X27 X68 X57 X84 X29 X43 X21 X53 X35 X11 X77 X49 X88 X67 X47 X19 X87 X44 X55 X71 X82 X26 X50 X10 X28 X85 X70 X45 X39 X66 X7 X30 X56 X64 X33 X89 X34 X79 X54 X90 X48 X60 X92 X32 X65 X94 X58 X3 X31 X2 X100 X98 X91 X9 X99 X1 X93 X8 X5 X97 X96 X95 X6 | X90 X73 X15 X40 X12 X14 X78 X18 X59 X87 X79 X57 X19 X72 X38 X26 X82 X16 X54 X74 X17 X77 X22 X10 X48 X60 X20 X51 X69 X86 X75 X30 X13 X46 X35 X61 X28 X85 X49 X42 X24 X80 X31 X70 X8 X45 X11 X76 X50 X62 X21 X84 X37 X9 X66 X88 X32 X68 X23 X71 X33 X7 X55 X89 X34 X5 X41 X58 X27 X52 X43 X29 X63 X81 X36 X83 X65 X39 X2 X25 X64 X53 X44 X67 X56 X93 X47 X91 X3 X4 X92 X97 X94 X100 X1 X95 X99 X96 X6 X98 | X66 X73 X15 X12 X79 X14 X22 X74 X19 X57 X30 X87 X48 X8 X17 X36 X86 X70 X9 X85 X33 X13 X55 X45 X23 X65 X39 X11 X88 X29 X81 X41 X53 X83 X5 X64 X44 X2 X93 X91 X100 X97 X4 X3 X94 X1 X99 X95 X6 X96 X98 |
| **count**                 | 100                                                          | 100                                                          | 51                                                           |
| **accuracy**              | 0.5007518                                                    | 0.5254848                                                    | <span style="color:red">**0.5264646**</span>                 |

* `Custom Data` 의 제작과정은 **5. Linear Regression**에서 `HV.model `이라는 데이터셋에서 나타남.

 <br> <br>     

### **5. Linear Regression**

<br>

class를 설명하는데 중요한 변수가 무엇이 있는지 파악하기 위하여 다중선형회귀를 진행하였음

```R
#변수 선별 함수 stepAIC() 사용
#모든 경우의 데이터셋에 대해서 가장 Adjusted R-squared 값과 p-value 값이 최적인 조합을 찾아내었음
newdata <- HV
newdata <- log(HV)
newdata <- HV.na.norm
newdata <- HV.log.norm
newdata <- HV.rm
newdata <- HV.rm.norm

newdata <- log.HV.rm.norm # 해당 데이터셋이 가장 최적이었음 (모든 전처리를 진행한 경우)

mod <- lm(class~., data = newdata)

mod.select <- stepAIC(mod)
mod.select
summary(mod.select)
```

```R
#stepAIC() final model
HV.model <- mod.select$model

HV.model$class <- as.factor(HV.model$class)
HV.model.raw <- mod.select$model
```

> **HV.model**
>
> * Object : 997
>
> * Variable : 52
>
> * Class Type : Factor *(Regressoin 진행시 Integer로 변환)*
>
> * Col
>
>   | x1      | **x2**  | **x3**   | **x4**    | **x5**  | **x6**  |
>   | ------- | ------- | -------- | --------- | ------- | ------- |
>   | **x8**  | **x9**  | **x11**  | **x12**   | **x13** | **x14** |
>   | **x15** | **x17** | **x19**  | **x22**   | **x23** | **x29** |
>   | **x30** | **x33** | **x36**  | **x39**   | **x41** | **x44** |
>   | **x45** | **x48** | **x53**  | **x55**   | **x57** | **x64** |
>   | **x65** | **x66** | **x70**  | **x73**   | **x74** | **x79** |
>   | **x81** | **x83** | **x85**  | **x86**   | **x87** | **x88** |
>   | **x91** | **x93** | **x94**  | **x95**   | **x96** | **x97** |
>   | **x98** | **x99** | **x100** | **class** |         |         |
>
> * Adjusted R-squared:  0.7103
>
> * p-value: < 2.2e-16
>
> * **(39 / 51)**의 비율로  영향력있는 변수라고 측정되었음 (* 이상 측정 기준)

<br><br>

<br>


### **6.  Logistic Regression**  

<br>

현재 Class의 분류 체계가 이진분류이기 때문에 Linear보다 Logistic Regression을 통해 모델평가를 진행하는 것이 더욱 효율적이라고 생각하여서 각 데이터셋에 대한 평가도 진행해보았으며 코드는 다음과 같다.

```R
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
```

<br>

***

<br>

대체로 구해본 데이터셋들에 대한 Logistic Regression에 대한 측정치는 아래와 같았다. 

| Raw Data                                     | Raw Data.Norm  | Log(Raw Data.Norm)                           |
| :------------------------------------------- | :------------- | :------------------------------------------- |
| 0.7021452                                    | 0.9604317      | 0.9694719                                    |
| **HV.RM**                                    | **HV.RM.Norm** | **Log(HV.RM.Norm)**                          |
| 0.7021063                                    | 0.7021063      | <span style="color:red">**0.9739218**</span> |
| **HV.Model**                                 |                |                                              |
| <span style="color:red">**0.9638917**</span> |                |                                              |

* *Log(HV.RM.Norm) = pre-processed dataset*

<br>

<br>

<br>



### **7.Model Evaluation ** 

<br>

모든 데이터들의 Accuracy 및 정상 작동을 확인하였으며, 이에 대한 모델들을 평가 검증하여, 정확도 대비 및 성능이 좋은 Feature Selection을 진행하기 위해 Evaluation을 진행하였다. 

다음은 Evaluation을 진행한 일부 코드이다. 

```R
#result model
#1. raw dataset
HV
lm1 <- lm(class~., data=HV)


#2. preprocessed dataset
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
```

<br>

| dataset                      | mRMR<br />raw dataset | mRMR<br />preprocessed dataset | <span style="color:red">mRMR<br />Custom HV model</span> |
| :--------------------------- | :--------------------- | :----------------------------- | :------------------------------------------------------- |
| **10-fold Accuracy**         | 0.5007518             | 0.525484                       | <span style="color:red">0.5264646</span>                 |
| **Adjusted <br />R-squared** | 0.1041                | 0.7013                         | <span style="color:red">0.7103 </span>                   |
| **p-value**                  | 6.812e-12             | < 2.2e-16                      | <span style="color:red">< 2.2e-16</span>                 |
| **Logistic Accuracy**        | 0.7021452             | 0.9739218                      | <span style="color:red">0.9638917</span>                 |

| dataset                      | hill climbing search | <span style="color:red">best first search</span> | backward search | forward search |
| :--------------------------- | :------------------- | :----------------------------------------------- | :-------------- | :------------- |
| **10-fold Accuracy**         | 0.5274848            | <span style="color:red">0.5395152</span>         | 0.5264848       | 0.5395152      |
| **Adjusted <br />R-squared** | 0.09625              | <span style="color:red">0.01495 </span>          | 0.1044          | 0.01495        |
| **p-value**                  | 2.981e-14            | <span style="color:red">0.0003111</span>         | 0.1044          | 0.0003111      |
| **Logistic Accuracy**        | 0.9488465            | <span style="color:red">0.6549649</span>         | 0.9739218       | 0.6549649      |

<br>

<br>

<br>



### **Conclusion**  

<br>  

결론적으로 가장 정확도를 보여주는 값으로는 다음의 Feature Selection의 결과가 적합하다고 판단하였다.   
<br>  

**(1)  1st Model**  : Pre-Processed Model with *log transformation & Normalizatoin*    
<br>  

> - **Selected Feature :** "X3"  "X6"  "X9"  "X16" "X90" 총 5개  
> <br>
> - **Using Method :** `Best First Search` (or `Forward Search`)   
> <br>
> - **Accuracy :** <span style="color:red">0.5395152</span>  
> <br>
> - **특이사항 :** `K-fold cross validation Accuracy `값은 가장 높으나 그 외의 값들은 <u>현저히 떨어지는 양상</u>을 나타내고 있음. 과도한 변수 생략으로 인한 것으로 유추됨  


<br>
<br>  

**(2)  2nd Model** : Custom Model with *stepAIC*  
<br>  

> - **Selected Feature :** "X1"    "X2"    "X3"    "X4"    "X5"    "X6"    "X8"    "X9"    "X11"   "X12"   "X13"   "X14"   "X15"   "X17"   "X19"   "X22"   "X23"   "X29"   "X30"   "X33"   "X36"   "X39"   "X41"   "X44"   "X45"   "X48"   "X53"   "X55"   "X57"   "X64"   "X65"   "X66"   "X70"   "X73"   "X74"   "X79"   "X81"   "X83"   "X85"   "X86"   "X87"   "X88"   "X91"   "X93"   "X94"   "X95"   "X96"   "X97"   "X98"   "X99"   "X100" 총 51개  
> <br>
> - **Using Method :** `mRMR` and `stepAIC()`  
> <br>
> - **Accuracy :** <span style="color:red">0.5264646</span>  
> <br>
> - **특이사항 :** 가장 높은 모델 설명도와 전반적으로 평이한 성능평가 결과를 나타내고 있음. 가장 높은 Accuracy를 표현하고 있지 않지만, 어느정도 신뢰도가 높은 모델로 판단이 됨                                                                               


<br>

즉, best first search 또는 forward search 를 통해서 추출한 (1)번의 Feature 조합이 가장 K-fold cross Validation에서 높은 Accuracy 를 보여주는 것으로 확인되었다고 결론내렸다.   
<br>  

그러나 과제에서 제안한 것 외에 여기서 (2)번을 함께 제안한 이유는 다음과 같다.   



> 1. 다중 선형회귀 함수 중 해당 데이터셋에서 가장 연관성이 높은 변수들의 조합을 자동으로 추출해주는 `stepAIC()` 함수를 기반으로 하였으므로 신뢰도가 다소 높음  
> 2. 해당 모델을 기반으로 다양한 방법의 evaluation을 진행하였을 때, 모두 평균이상의 결과가 나왔음  
> 3. 특히 (1)번의 모델의 경우 `Adjusted R-squared`값이 *<u>0.01495</u>* 밖에 되지 않는다는 것은 설명력이 1%밖에 미치지 못한다는 것인데, 이에 반해 (2)번의 모델의 경우 <span style="color:red">0.7103 </span>으로 71%의 높은 설득력을 가지고 있음  
> 4. 또한 이진분류의 성향을 띈 데이터셋이므로 `Logistic Regression` 에서 비롯된 `Accuracy`값이 2번째로 높은데, 이러한 경우에 오히려 (2)번 모델이 더욱 높은 성능을 낼 수도 있음  

<br>  



지금까지 Hill_Valley Dataset을 기반으로 본 시험과제를 진행하였고, 이를 위한 연구를 통해 데이터의 전처리, 그리고 효율적으로 자원과 시간을 사용하기 위한 Feature Selection을 마무리할 수 있었다.   
다만 아쉬운 점이 있다면 본 과제를 진행하면서 몇 가지 놓친 점들이 있어 생각보다 시간을 많이 지체하게 되었다.   

<br>  

<br>  

#### **(1) Exhausted Search**  

<br>  


우선 첫번째로 아쉬웠던 점은 가장 비효율적이어도 높은 정확도를 찾아낼 수 있는 exhuasted search 방법을 사용하지 못했다는 점이다. 모든 부분집합에 대한 정확도평가를 측정하는 부분에 있어 매우 비효율적이라는 부분이 있지만, 정확하게 진행할 수 있다는 점이 해당 알고리즘의 가장 큰 장점인데, 그렇게 된다면 기존의 dataset의 경우 `2^101`의 부분집합의 개수만큼 연산을 해야한다고 한다. <span style="color:orange">(참고로 우주 전체 원자 수가 10^80이라는데 우주 전체 원자 수의 40%정도됨)</span>
이에 대해서 처음에 아무런 자각도 하지 않고 열심히 노트북을 혹사시켰다가 이틀이 지나도 끝이 나지 않는 것을 보고나서 *'아차'* 하는 마음에 조사를 해 보았고, 얼마나 안일하게 스타트를 끊었는가를 알아차리게 되었다. 

<br>

<p align="center"><img src="https://user-images.githubusercontent.com/19165180/121446031-a3e36700-c9cd-11eb-8710-b0e0a51626b4.png" alt="image" style="zoom: 33%;" />                     <img src="https://user-images.githubusercontent.com/19165180/121446192-f6bd1e80-c9cd-11eb-8161-dcf95b1f57e6.png" alt="image" style="zoom: 33%;" />  </p>


> Exhausted Search 코드가 이틀이 지나도 안끝나서 무엇이 잘못되었다는 것을 깨달았을 때 나누었던 대화.  

<br>

<br>

<br>

<br>

<br>

#### **(2) Accuracy = 0...?**

<br>  


결국 전처리 과정부터 열심히 진행을 한 후, 큰 마음을 먹고  `exhausted search`는 과감하게 제외하고 다시`FSelection` 코드를 돌렸는데 이제 또다른 난관으로는 결과값에서 <span style="color:red">**`acc=0`**</span>이라는 글씨가 뜨기 시작했다.  

<br>  

<p align="center"><img src="https://user-images.githubusercontent.com/19165180/121447049-d7bf8c00-c9cf-11eb-8910-edada0d796ea.png" alt="image" style="zoom: 50%;" />  </p>


>함께 당황하고 있는 다른 연구실의 외국인 수강생  



<br>

<Br>



어떤 수를 쓰고, 새로 함수를 짜고해도 안되었고, SVM 모델이 잘못된건지, 교수님께서 주신 K-fold cross validation 함수를 수정해야하는건지, 아니면 내가 다시 customizing 해야하는건지 갈피를 정말 전혀 잡지도 못하고 과제 마감 D-2까지 Accuracy 측정을 하지 못하여서 계속 Adjusted R-squared 값과 P-value, RMSE, MAE 이런 단편적인 값들만 계속 찾아내고 있어서 답답할 따름이었다.   
그런데 불현듯 떠오르는 간단한 내용이 하나가 떠올랐고, 바로 `Class ` 변수의 Type을 확인하러 갔는데 역시나.   Factor타입이 아닌 Integer 타입이었고, 이를 Factor 로 형변환을 해주고 나니까 결과값이 나와서 과제를 마무리 할 수 있었다.   

<br>

<p align="center"><img src="https://user-images.githubusercontent.com/19165180/121448500-c166ff80-c9d2-11eb-9349-6acefd60fbd8.png" alt="image" style="zoom: 50%;" />  </p>

> 너무나도 당연하게 Class 의 Type이 Factor라고 생각한 것 같다  
> 앞으로는 당연하게 넘기는 것이 아닌, 데이터를 받으면 당연해보이는 것들도 반드시 확인하는 습관을 들이도록 해야겠다고 느꼈다.  

<br>

<br>

<br>

<br>

<br>

#### **(3) 이렇게 많은 변수들에 대해서 Outlier를 어떻게 그려내야 하는가**

<br>  


전처리 과정에서 처음에 가장 어려웠던 부분은 Outlier를 한꺼번에 어떻게 정리해야하는지였다.  대체로 Outlier를 처리하는 방법으로는 거의 30개 미만의 적은 변수들을 통해서 `boxplot()`을 기반으로 이상치를 걷어내고 남은 데이터들에 대해 정제 후 다시 데이터분석을 진행하는데, 애초에 변수가 너무 많아 쪼개서 진행하기도 어려웠고, `plot()`을 그려도 제대로 뭔가 표현이 되지 않고 오류가 떴다.  

(대체로 오류는 주변 여백 마진이 남지 않거나, 데이터의 양이 너무 방대해서 그리기 어렵다는 말이 었다.)  

아직 이 부분에 대한 확실한 답안을 얻은 것은 아니지만, 우선 `for`문을 활용하여 각 `boxplot()`을 그리지는 않고 열 별로 `index`만 추출해낼 수 있도록 연산하고, 해당 `index`의 행을 제거하는 방식으로 진행하였는데, 이에 대한 모범답안이 있거나, 다른 방법이 있다면 공유를 받거나 학습해보고 싶다고 생각이 들었다.   

<br>

#### **R codes** 

본 과제의 소스코드는 `Github`에 모두 공개되어 있으며, 소스코드 및 사용한 데이터셋을 모두 확인할 수 있습니다.  

   <br>

* **Github URL : <https://github.com/wpsl94/DataMining_Final-2021>**  

**<br>**  

<br>

<br>

<br>

<br>

