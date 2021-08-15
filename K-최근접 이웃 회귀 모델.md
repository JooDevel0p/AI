## K-최근접 이웃 회귀 모델
----------
### 길이만 가지고 무게 예측하기

#### >농어 데이터 저장하기
```
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

#### >농어 데이터 분산도로 나타내기
```
plt.scatter(perch_length,perch_weight)
plt.show()
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLygew%2FbtrbGAW6Tyd%2FS36rSGqM8KP0GAElPPkaJk%2Fimg.png" width="400px" height="300px" ></img>

#### >라이브러리를 이용하여 정확한 비율로 데이터 나누기

```
>from sklearn.model_selection import train_test_split
>train_input,test_input,train_target,test_target=train_test_split(perch_length,perch_weight)
>
>train_input 

array([27.3, 37. , 43.5, 44. , 19.6, 21.3, 23.5, 25. , 40. , 22. , 28.7,
       19. , 22. , 40. , 18.7, 32.8, 24.6, 40. , 15. , 36. , 39. , 34.5,
       20. , 22. , 18. , 39. ,  8.4, 26.5, 27.5, 35. , 24. , 22. , 21. ,
       36.5, 23. , 16.2, 22. , 21. , 40. , 27.5, 43. , 22.5])
```
##### -데이터를 나누는 default 값은 75%:훈련 데이터 , 25%:시험 데이터


#### >문제지에 해당하는 1차원 넘파이 배열인 훈련 데이터를 2차원 넘파이 배열로 바꿔주기

```
train_input=train_input.reshape(-1,1)
```
##### -reshape를 다시 바꾸는 것

##### * -1 : 2차원 넘파이 배열

 ##### * 1 : 요인의 개수

#### >시험 데이터의 문제지 2차원 넘파이 배열로 바꾸기

```
test_input=test_input.reshape(-1,1)
```

#### >학습 데이터의 문제지 조회하기
```
>train_input

array([[27.3],
       [37. ],
       [43.5],
       [44. ],
       [21.3],
.....생략.....
       [21. ],
       [40. ],
       [27.5],
       [43. ],
       [22.5]])
````
##### -2차원 넘파이 배열로 바뀐 것을 확인할 수 있다.


#### >회귀 라이브러리를 통해 훈련 데이터로 학습시키기

```
>from sklearn.neighbors import KNeighborsRegressor
>
>knr=KNeighborsRegressor()
>knr.fit(train_input,train_target)

KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                    weights='uniform')
```


#### >모델 성능 평가하기

```
>knr.score(test_input,test_target)

0.9319917476479095
```
##### -맞추는 것에 중점을 두는 것이 아니라 얼마나 정답에 가깝냐에 중점을 둔다.


#### >무게 예측해보기
```
>knr.predict(test_input[:8])

array([950.  , 123.  , 950.  , 206.2 , 268.6 , 327.8 ,  55.08, 129.  ])
```

#### >실제값 조회해보기
```
>test_target[:8]

array([1100.,  150., 1100.,  197.,  250.,  320.,   32.,  145.])
 
 ```
 
#### >평균적 오차 구하기
```
>np.mean(abs(knr.predict(test_input[:8])-test_target[:8])) 

50.21
```

#### >훈련데이터로 학습하고 훈련데이터와 시험데이터를 넣었을 때의 성능 출력하기
```
>knr.fit(train_input,train_target)   
>
>print(knr.score(train_input,train_target))
>print(knr.score(test_input,test_target))

0.9793711240304775
0.9319917476479095
```
##### -당연히 훈련데이터를 넣었을 때 높아야한다.

#### >농어의 길이로 무게 예측해보기
```
knr=KNeighborsRegressor()

x=np.arange(5,45).reshape(-1,1)

for n in ([1,5,10,20]):
  knr.n_neighbors=n
  knr.fit(train_input,train_target)

  prediction=knr.predict(x)

  plt.scatter(train_input,train_target)
  plt.plot(x,prediction,'r')
  plt.show()
  ```
##### -농어 길이 5~44까지의 무게 예측해 보기

##### -1,5,10,20 으로 평균낼 개수 설정하기

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSINaC%2FbtrbCtYxBsa%2FoNq19u6jL8NseSGzDckif0%2Fimg.png" width="400px" height="300px" ></img>

n=1

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcBqiRp%2FbtrbIHBsJLb%2FTT7Y8bVhs3x3AIF8nGxbtK%2Fimg.png" width="400px" height="300px" ></img>

n=5

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIZuc8%2FbtrbCs6m8X2%2FZwSUkpoN4uZdGbNU1dko4K%2Fimg.png" width="400px" height="300px" ></img>

n=10

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FC26Vh%2FbtrbwhxOwP3%2FHtTcb0fi6pZPY9BnMFkZdK%2Fimg.png" width="400px" height="300px" ></img>

n=20
##### -적당한 개수를 잡는 것이 중요하다. 너무 적으면 과대적합, 많으면 과소적합이 일어난다.

#### >최근접 3개로 평균내고 데이터 학습시키기
```
>knr=KNeighborsRegressor(n_neighbors=3)
>knr.fit(train_input,train_target)


KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                    weights='uniform')
```


#### >길이로 무게 예측하기
```
>knr.predict([[50]]) 

array([1000.])

```
##### -길이가 50인 생선의 무게 예측하기

#### >길이가 50일때의 가장 가까운 데이터 3개 산점도에 나타내기

```
distances,indexes=knr.kneighbors([[50]])

plt.scatter(train_input,train_target)
plt.scatter(train_input[indexes],train_target[indexes],marker='D')
plt.scatter(50,1033,marker='^')
plt.show()
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkMQSW%2FbtrbAA4F3ae%2Fn5QSinzKrPkgzZ4nTqjNsK%2Fimg.png" width="400px" height="300px" ></img>


#### >K-최근접 모델의 한계점
```
distances,indexes=knr.kneighbors([[500]])  

plt.scatter(train_input,train_target)
plt.scatter(train_input[indexes],train_target[indexes],marker='D')
plt.scatter(500,1033,marker='^')
plt.show()

```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FczAHrv%2FbtrbIchyt2r%2Fa8W2QQKZ5u27vWkh9cKgW1%2Fimg.png" width="400px" height="300px" ></img>


##### -길이 500을 넣어도 50을 넣었을 때와 같은 무게로 예측한다.


