## 선형 회귀란?? 데이터들간의 상관 관계를 최적의 직선으로 나타내어 입력변수를 직선의 방정식에 대입시켜 목표변수를 예측하는 방법

<img src="https://mblogthumb-phinf.pstatic.net/MjAyMDAxMjhfMTc1/MDAxNTgwMTkxMDM3ODc3.YkT5o8SO8DfOhAxn4zh-0Ncy89ii_x8F50tzd0SQsBsg.MqMhHI2SfhdKMdJekUVRdI_bCHJFlVI6NDwFtKiWJjYg.JPEG.synviva/Capture.JPG?type=w800" width="300px" height="200px" ></img>
---------

#### >농어 데이터 저장하기
```

import numpy as np


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
       
```


#### >저장한 데이터 산점도로 나타내기
```
import matplotlib.pyplot as plt
     

plt.scatter(perch_length,perch_weight)
plt.show()

```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb7Q9vk%2FbtrbENWQYmO%2F1P3itQP7u8Ykhk61fnkZP0%2Fimg.png" width="300px" height="200px" ></img>
##### -선의 방적식을 찾으면 다른 값도 예측 가능하다.

#### >선형 회귀 라이브러리를 데이터 학습시키기
```
>from sklearn.linear_model import LinearRegression 
>
>lr=LinearRegression()
>lr.fit(train_input,train_target)


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```
##### -선을 찾았을 것이다.


#### >선의 기울기와 절편 출력하기

```
>lr.coef_           #기울기

array([39.51700071])


>lr.intercept_      #절편

-747.0177226646915
```


#### >길이가 50일 때 길이 예측해보기
```
>lr.predict([[50]])

array([1228.83231267])
```


#### >찾은 방정식과 분산도 그리기
```
plt.scatter(train_input,train_target)
plt.plot([15,50],[15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_],'r')
plt.scatter(50,1228,marker='^')
plt.show()

```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbaSZvR%2Fbtrbvx8lyuP%2FKSkvGLE7JBsmIrtMPfxlp1%2Fimg.png" width="300px" height="200px" ></img>

##### -축의 범위는 농어의 길이로 15~50까지 , y축의 범위는 직선의 방정식인 15*기울기+절편~50*기울기+절편까지
##### -요인이 하나이기때문에 직선의 방정식이 만들어진다.


#### >데이터 늘리기

```
train_poly=np.column_stack([train_input**2,train_input])
test_poly=np.column_stack([test_input**2,test_input])
```
##### -곡선을 만들기 위해서는 더 많은 요인이 필요하다.
##### -농어의 길이 데이터 밖에 없는데 요인을 늘리기 위해 길이*2를 두 번째 데이터로 만든다면???이 데이터를 추가하면 요인을 2개로 볼까?? 다른 데이터로 보지 않는다.
##### -제곱을 넣으면 컴퓨터가 다른 데이터로 본다.
##### -길이^2+길이 는 다른 성격의 요인이라 생각한다.(추천하는 방법은 아니다.)
##### -column_stack은 2차원 넘 파이 배열로 데이터를 한 번에 만들어준다.

#### >데이터 출력하기
```
>train_poly #2차원 넘파이 배열로


array([[ 756.25,   27.5 ],
       [ 441.  ,   21.  ],
       [ 484.  ,   22.  ],
       [ 441.  ,   21.  ],
       [ 361.  ,   19.  ],
       [ 702.25,   26.5 ],
      .......생략........
       [1936.  ,   44.  ],
       [1369.  ,   37.  ],
       [ 484.  ,   22.  ],
       [1521.  ,   39.  ],
       [1600.  ,   40.  ],
       [1521.  ,   39.  ]])
```
##### -[길이^2, 길이]

#### >데이터 학습시키기
```
>lr=LinearRegression()    
>lr.fit(train_poly,train_target)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```

#### >길이로 무게 예측하기
```
>lr.predict([[50**2,50]])

array([1509.3376394])
```

#### >기울기 출력하기
```
>lr.coef_ 

array([  0.93580772, -18.16426852])
```
##### -요인이 2개니까 기울기도 2개


#### >절편 출력하기
```
>lr.intercept_

78.03177452291123
```
##### -무게=0.93580772*길이^2-18.16426852*길이+78.03177452291123



#### >곡선과 산점도 나타내고 예측하고자 하는 데이터 나타내기
```
point=np.arange(15,50)

plt.scatter(train_input,train_target)
plt.plot(point,0.9*point**2-18.2*point+78.0,'r')
plt.scatter(50,1509.3,marker='^')     #길이,무게
plt.show()
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbMjvTu%2Fbtrbq4y6rGs%2F0As1orFS4E7838IDQ0UKz1%2Fimg.png" width="300px" height="200px" ></img>


#### >훈련 데이터와 시험 데이터를 넣었을 때의 성능 알아보기
```
>print(lr.score(train_poly,train_target)) 
>print(lr.score(test_poly,test_target))

0.9664895223860305
0.9810636887246954
```
##### -훈련 데이터로 학습을 한 모델에 훈련 데이터와 시험 데이터를 넣어 본 결과이다.
##### 당연히 훈련 데이터를 넣었을 때의 정확도가 더 놓아야 하는데 시험 데이터를 넣었을 때의 정확도가 더 높게 나왔다.
##### 왜일까?? 과소 적합 현상
##### + 과대 적합 : 훈련 데이터에 너무 적응되서 훈련 데이터의 정확도는 높으나, 테스트 데이터의 정확도는 낮은 경우
##### + 과소 적합 : 훈련이 덜 되었거나 문제가 너무 쉬워서 훈련시킨 데이터보다 시험 데이터의 정확도가 높은 경우


### 과소 적합된 모델 더 복잡하게 만들기


#### >데이터 3개 넘파이 배열에 저장하기

```
>import pandas as pd
>
>perch_full=pd.read_csv("https://bit.ly/perch_csv").to_numpy()  
>perch_full

array([[ 8.4 ,  2.11,  1.41],
       [13.7 ,  3.53,  2.  ],
       [15.  ,  3.82,  2.43],
       [16.2 ,  4.59,  2.63],
       [17.4 ,  4.59,  2.94],
       [18.  ,  5.22,  3.32],
       [18.7 ,  5.2 ,  3.12],
       .......생략...........
       [40.  , 11.14,  6.63],
       [42.  , 12.8 ,  6.87],
       [43.  , 11.93,  7.28],
       [43.  , 12.51,  7.42],
       [43.5 , 12.6 ,  8.14],
       [44.  , 12.49,  7.6 ]])

```

#### >데이터 일정한 비율로 나누어 저장하기
```
train_input,test_input,train_target,test_target=train_test_split(perch_full,perch_weight)
 
 ```
 
#### >요인 강제로 늘리기 위한 학습시키기
```
>from sklearn.preprocessing import PolynomialFeatures
>
>poly=PolynomialFeatures()
>poly.fit([[2,3]]) 
PolynomialFeatures(degree=2, include_bias=True, interaction_only=False,
                   order='C')
```

##### -2,3을 가지고 만들 수 있는 요인을 모두 찾는다.
##### -1은 기본 값, 2,3,4,9,6

#### >만들 수 있는 모든 요인 만들기
```
>poly.transform([[2,3]]) #제곱까지만

array([[1., 2., 3., 4., 6., 9.]])

```

#### >라이브러리를 이용하여 훈련 데이터로 만들 수 있는 모든 요인 만들기
```
poly=PolynomialFeatures()
poly.fit(train_input) 

train_poly=poly.transform(train_input)   
test_poly=poly.transform(test_input) 
```

#### >요인 몇 개 인지 알아보기
```
>train_poly.shape

(42, 10)
```
##### -3개였던 요인이 10개로 늘어났다.

#### >늘어난 데이터로 학습시키기
```
>lr=LinearRegression() 
>lr.fit(train_poly,train_target)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

```
#### >성능 알아보기

```
>print(lr.score(train_poly, train_target))   
>print(lr.score(test_poly,test_target))

0.9943428614865089
0.9512175178727978

```
##### -과소적합을 막아주고 잘 학습되었다.


#### >요인 늘리는 조건 바꾸기

```
poly=PolynomialFeatures(degree=5) 
poly.fit(train_input)  

train_poly=poly.transform(train_input)   
test_poly=poly.transform(test_input) 

```
##### -degree=5는 5 제곱까지도 허용한다는 것을 의미한다.


#### >요인 개수 확인하기
```
>train_poly.shape 

(42, 56)

```
##### -요인이 10개에서 56개로 늘어난 것을 확인할 수 있다.




#### >학습시키기

```
>lr=LinearRegression()   
>lr.fit(train_poly,train_target)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```


#### >성능 알아보기
```
>print(lr.score(train_poly, train_target))  
>print(lr.score(test_poly,test_target))   


0.9999999999999992
-17453.176386077957
```
##### -훈련 데이터를 넣었을 때는 만점에 가까운 정답률을 보이는데 테스트 데이터를 넣으니까 거의 다 틀린 것을 볼 수 있다. 그 이유는 훈련 데이터에 과대 적합되었기 때문이다.
##### -음수가 나왔다는 것은 다 틀렸다는 것이고 발산했다는 것을 의미한다.

### 과대 적합 막아보기

#### >표준화 라이브러리를 통해 표준화시키기
```
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_poly)        

train_scaled=ss.transform(train_poly) 
test_scaled=ss.transform(test_poly)
```
##### -train_scaled에 train_poly에 평균을 빼고 표준편차로 나눠준 값을 넣어준다.

##### test_scaled에 test_poly에 평균을 빼고 표준편차로 나눠준 값을 넣어준다.



#### >학습시키기
```
>lr=LinearRegression()
>lr.fit(train_poly,train_target)
>
>print(lr.score(train_poly, train_target)) 
>print(lr.score(test_poly,test_target)) 

0.9999999999999992
-17453.176386077957

```

##### -표준화를 해줘도 요인이 너무 많아서 과대 적합된 것을 확인할 수 있다.

##### -과대 적합을 막는다는 것은 완벽하게 학습되는 것을 막는 것이다.

##### LinearRegression()+규제 기능이 추가되는 라이브러리를 사용하여 과대 적합을 막는다.
##### 여기서 규제란?? 완벽하게 학습되는 것을 막는 것
##### 규제 : 릿지, 라쏘
+ 릿지 : 요인 앞에 가중치가 곱해지는데 제곱해서 가중치를 낮추기
+ 라쏘 : 요인 가중치를 절댓값을 해서 낮추는 기능



#### >릿지를 이용하여 데이터 학습시키기
```
>from sklearn.linear_model import Ridge
>
>ridge=Ridge()
>ridge.fit(train_scaled,train_target)


Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
```

#### >릿지 사용하여 학습시킨 모델 성능 알아보기
```
>print(ridge.score(train_scaled, train_target))  
>print(ridge.score(test_scaled,test_target)) 


0.9914714252079576
0.9689786483726737
```
##### -과대 적합을 막은 것을 확인할 수 있다.


#### >규제의 강도 조절하고 강도 별 모델 성능 알아보기

```
train_score=[]
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]      #규제 강도

for i in alpha_list:
  ridge=Ridge(alpha=i)
  ridge.fit(train_scaled,train_target)

  train_score.append(ridge.score(train_scaled,train_target))
  test_score.append(ridge.score(test_scaled,test_target))
```


#### >강도 별 성능 그래프로 알아보기
```
plt.plot(np.log10(alpha_list),train_score) 
plt.plot(np.log10(alpha_list),test_score)
plt.show()
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcw0LQD%2FbtrbABv9rYL%2FszY8hlzdvUQ65PmcmHKOgk%2Fimg.png" width="300px" height="200px" ></img>

##### -파란색은 훈련 데이터, 주황색은 테스트 데이터

##### -테스트 데이터 그래프가 훈련 데이터 그래프보다 아래 있으면서 차이가 가장 작은 구간이 가장 성능이 좋은 구간이다.

##### -x축이 1 일 때 가장 좋은 것으로 보인다.



#### >라쏘를 이용하여 학습시키기
```
>from sklearn.linear_model import Lasso 
>lasso=Lasso()
>lasso.fit(train_scaled,train_target)  #학습



/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 703.4053255411854, tolerance: 487.37066478571427
  positive)
Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='cyclic', tol=0.0001, warm_start=False)
 
 ```
 
#### >성능 알아보기
```
>print(lasso.score(train_scaled,train_target))
>print(lasso.score(test_scaled,test_target))

0.9899267121214844
0.9720638509745514
```

##### -과소 적합을 확인할 수 있다.


#### >규제 강도에 따른 성능 알아보기
```
>train_score=[]
>test_score=[]
>
>alpha_list=[0.001,0.01,0.1,1,10,100]
>
>for i in alpha_list:
>  lasso=Lasso(alpha=i)
>  lasso.fit(train_scaled,train_target)
>
>  train_score.append(lasso.score(train_scaled,train_target))
>  test_score.append(lasso.score(test_scaled,test_target)) 


 /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 12409.095481574443, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 12497.004653629785, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8667.22283542711, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 703.4053255411854, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 732.8452389490412, tolerance: 487.37066478571427
  positive)
```

##### -출력물은 규제를 하는데 학습량이 부족하다는 것을 의미한다.

#### >학습량을 늘려서 다시 성능 알아보기

```
>train_score=[]
>test_score=[]
>
>alpha_list=[0.001,0.01,0.1,1,10,100]
>
>for i in alpha_list:
>  lasso=Lasso(alpha=i,max_iter=10000) 
>  lasso.fit(train_scaled,train_target)
>
>  train_score.append(lasso.score(train_scaled,train_target))
>  test_score.append(lasso.score(test_scaled,test_target)) 
  
  
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 7374.122665484362, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 5449.021095998551, tolerance: 487.37066478571427
  positive)
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:476: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 1637.373438096416, tolerance: 487.37066478571427
  positive)
```

##### -max_iter=10000을 통해서 학습량을 10000번으로 늘린다.

##### -그래도 학습량이 부족하다고 나올 수 있다.


#### >규제 강도별 성능 그래프로 알아보기
```
plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.show()

```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fl5zFI%2FbtrbKF4F3zq%2FaNJ1n2QjXFCKiCTgikbeNK%2Fimg.png" width="300px" height="200px" ></img>

##### -파란색은 훈련 데이터, 주황색은 테스트 데이터이다.
##### - 1일 때 가장 좋아 보인다.


#### >컴퓨터가 요인을 몇 개 가져다 썼는지 알아보기
```
lasso.coef_ 

array([  0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   4.79963419,   7.48874254,  39.07527353,
         0.        ,  36.2029221 , 111.57366905,  84.10713138,
         0.        ,  51.19629676,   0.        ,   0.        ,
        16.99602473,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,  -0.        ,  -0.        ,   0.        ,
         0.        ,  -0.        ,   0.        ,  -0.        ,
        -0.        ,   0.        ,  -0.        ,  -0.        ,
        -0.        ,   0.        ,  -0.        ,  -0.        ,
        -0.        ,  -0.        ,   0.        ,  -0.        ,
        -0.        ,  -0.        ,  -0.        , -12.75206691])
```
##### -0 은 가져다 쓰지 않은 것을 의미한다.

#### >쓰지 않은 데이터 개수 알아보기
```
>np.sum(lasso.coef_==0)

47

```
##### -56-47=9개 데이터 사용한 것을 알 수 있다.

### 생선 종류 예측해보기

#### >데이터 가져오기
```
import pandas as pd

fish=pd.read_csv("https://bit.ly/fish_csv")
fish

```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbdd4L6%2FbtrbABJHwmO%2FLVLuJ103kltre6AhkLTn81%2Fimg.png" width="400px" height="300px" ></img>

#### >생선의 종류 알아보기
```
>np.unique(fish['Species'])

array(['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'],
      dtype=object)
```
##### -생선이 7가지 종류가 있는 것을 확인할 수 있다.

#### >문제지, 정답지 만들기
```
fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target=fish['Species'].to_numpy()
```
##### -어떤 알고리즘을 사용하면 좋을까?? k-최근접 이웃 분류 알고리즘

 
#### >일정한 비율로 데이터 나누기
```
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target)
````

#### >데이터 표준화시키기
```
ss=StandardScaler()
ss.fit(train_input)

train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)
 ```
 
#### >학습시키기
```
>from sklearn.neighbors import KNeighborsClassifier
>
>kn=KNeighborsClassifier()
>kn.fit(train_scaled,train_target)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
 ```
 
#### >성능 알아보기
```
>print(kn.score(train_scaled,train_target))
>print(kn.score(test_scaled,test_target))

0.8235294117647058
0.825
```
##### -과소 적합된 느낌이 있다. 둘 다 학습이 덜 되어 확률이 낮기 때문이다.

#### >예측해보기
```
>kn.predict(test_scaled[:5])

array(['Parkki', 'Parkki', 'Perch', 'Bream', 'Bream'], dtype=object)

```

#### >예측해보기
```
test_target[:5]

array(['Parkki', 'Parkki', 'Perch', 'Bream', 'Bream'], dtype=object)
```

#### >어떻게 예측했는지 알아보기
```
>proba=kn.predict_proba(test_scaled[:5])
>np.round(proba,decimals=3)


array([[0. , 0. , 0.4, 0.6, 0. , 0. , 0. ],
       [0.6, 0.4, 0. , 0. , 0. , 0. , 0. ],
       [1. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 1. , 0. ],
       [0. , 0. , 0.8, 0. , 0. , 0. , 0.2]])
```
##### -반올림하여 3자리까지 나타냈다.


#### >컴퓨터가 예측한 생선 알아보기

```
>kn.classes_ 


array(['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'],
      dtype=object)
```
--------------
## 로지스틱 회귀란?? 시그모이드 함수의 최적 선을 찾고 시그모이드 함수의 0~1 사이의 값 중에 클래스에 분류될 확률을 가지고 분류 값을 결정하는 것

<img src="https://postfiles.pstatic.net/MjAyMTAxMDZfMTg5/MDAxNjA5OTE2MTc1NDUx.OnJ4-Pwciqjh8h9p-fl0kGrG3ZpXyDzuLEsHCVrcyoog.S5ThG8NFkvgJkpWmMhx0l_iQNLmhg9LrElUjdNlelbgg.PNG.ehdgus8725/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-01-06_%EC%98%A4%ED%9B%84_3.55.51.png?type=w773" width="400px" height="300px" ></img>

### 로지스틱 회귀 사용하기

#### >로지스틱 라이브러리를 이용하여 학습시키기
```
>from sklearn.linear_model import LogisticRegression
>
>lr=LogisticRegression()
>lr.fit(train_scaled,train_target)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
 
 ```
 

#### >성능 알아보기
```
>print(lr.score(train_scaled,train_target))
>print(lr.score(test_scaled,test_target))

0.8319327731092437
0.8
```

#### >예측해보기
```
>lr.predict(test_scaled[:5])

array(['Pike', 'Bream', 'Bream', 'Smelt', 'Perch'], dtype=object)
 
 ```

#### >실제 정답 알아보기
```
>test_target[:5] 

array(['Pike', 'Bream', 'Bream', 'Smelt', 'Perch'], dtype=object)
```
##### -예측 값이 모두 맞은 것을 확인할 수 있다.


#### >확률로 알아보기
```
>lr.predict_log_proba(test_scaled[:5])
>np.round(proba,decimals=3)


array([[0. , 0. , 0.4, 0.6, 0. , 0. , 0. ],
       [0.6, 0.4, 0. , 0. , 0. , 0. , 0. ],
       [1. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 1. , 0. ],
       [0. , 0. , 0.8, 0. , 0. , 0. , 0.2]])
```


