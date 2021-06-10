---

*Title: "Data Mining(2021)-Final Report"*  
*Author: "72200117 Kim Hee Ju"*  
*Due date: '2021 6 10 '*  

------



<br>  

#### **👏 시작하기 전에..** 

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

### **[Assignment]** 💡

**Hill_Vally 데이터셋에 대해 가장 높은 분류 정확도를 제공하는 Feature의 집합과 Accuracy를 제시하시오**     

> * Hill_vally 데이터셋의 마지막 컬럼이 class label 임
> * Accuracy 의 평가는 10-fold cross validation 으로 하며, seed 값은 100 으로 함
> * 전처리 후 작업 진행
>   <br>

<br>

<br>

### **1. Prepare for Data Analysis**  📊

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
    

### **2. Data Pre-Processing** ⛏

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

<br>

<br>
<br>

#### **(2) Outlier detection and Elimination**    

표준화를 진행하여 데이터셋에서 얼마나 동떨어진 데이터들이 있는 지(=`Outlier`)를 살펴본 후, 이를 제거해주는 작업을 진행하였다.  대체로 평균에서 매우 먼 값이며, 표준편차보다는 큰 값을 의미한다. 통상적으로 신뢰구간 밖의 범위의 값들을 Outlier로 정의하고, 해당 인덱스의 값들은 제외하거나 평균, 혹은 중앙값으로 대체하여 model을 재구성한다. 

또한 이상치로 인해 <u>**나머지 값들은 0에 가까운 숫자로 왜곡되게 정규화될 가능성이 높기 때문에**</u> 이상치 제거 과정을 거친 후 정규화 작업을 진행하도록 결정하였다.

본 실험에서는 후에 탐색 시간을 줄이기 위해 데이터의 수를 줄이기 위하여 대체보다 ***탐색 후 제거***를 하는 방향으로 진행하였다.   

<br><br>
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



<br>

***

<br>
<br><br>

### **3. k-fold Cross Validation** 🥞

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

<br>

<br>

### **4. Feature Selection** ✍

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

가장 정제된 데이터셋인 `log.HV.rm.norm` 데이터셋을 기반으로 기본적으로 4가지의 algorithm을 기준으로 FS를 진행함

`log.HV.rm.norm` : Outlier Removed Hill_Valley Dataset with Normalization and Log Transforming

> - hill climbing search
> - best firsth search
> - backward search
> - forward search

단, exhausted search 의 경우 시간 소요가 너무 오래 걸려 제외함
<br>  

#### **(2) mRMR** 

Raw Dataset, Pre-processed Dataset, Customized Dataset 총 세 가지에 대한 FS를 진행함

 <br> <br>     

### **5. Linear Regression** 🧶

<br>

class를 설명하는데 중요한 변수가 무엇이 있는지 파악하기 위하여 다중선형회귀를 진행하였음

<br><br>

<br>


### **6.  Logistic Regression**  💊

<br>

현재 Class의 분류 체계가 이진분류이기 때문에 Linear보다 Logistic Regression을 통해 모델평가를 진행하는 것이 더욱 효율적이라고 생각하여서 각 데이터셋에 대한 평가도 진행함.

<br>



### **7.Model Evaluation** 🎲

<br>

모든 데이터들의 Accuracy 및 정상 작동을 확인하였으며, 이에 대한 모델들을 평가 검증하여, 정확도 대비 및 성능이 좋은 Feature Selection을 진행하기 위해 Evaluation을 진행하였다. 



<br>



