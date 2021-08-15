## 로또 최신 회차까지 데이터 가져오기
------




##### >url 가져올 때 메인창에서 검색을 해야하는 이유
+ 메인창에서 검색

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fej2TCz%2Fbtq9Y56pZ8X%2FoKWb8c9xkmIU0tSU0gl1h0%2Fimg.png" width="400px" height="100px" title="메인창 검색" alt="메인창 검색"></img>
+ 메인창에서 '로또' 검색한 url

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F97Yrb%2Fbtq9PoS6eoI%2F8gYdDYRf6lDrRk2MOmePG0%2Fimg.png" width="400px" height="100px" title="메인창에서 '로또' 검색한 url" alt="메인창에서 '로또' 검색한 url"></img>
+ '날씨'를 검색하고 연속으로 '로또'를 검색

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkLlES%2Fbtq9R5M4tAM%2Fxf58ZMhh762yNXRl6iIZB1%2Fimg.png" width="400px" height="100px" title="날씨'를 검색하고 연속으로 '로또'를 검색" alt="'날씨'를 검색하고 연속으로 '로또'를 검색"></img>

+ '날씨' 검색 후 '로또' 검색한 url

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbN8z5z%2Fbtq92ysX1Hn%2FgnmUjqhKjdEqT9FcMatnkK%2Fimg.png" width="400px" height="90px" title="'날씨' 검색 후 '로또' 검색한 url" alt="'날씨'를 검색하고 연속으로 '로또'를 검색한 url"></img>

***어떤 키워드를 검색하고 메인 창이 아닌 창에서 연속으로 다른 키워드를 검색할 경우, url에 기록 남기 때문에, 깔끔한 url 을 가져오기 위해 메인 창에서 검색해야 한다.***


##### >로또 현재 회차 데이터 가져오기

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FRtdTO%2Fbtq9Xbluh6M%2F7vmO0bxgKDxMZTIyLAwHY1%2Fimg.png" width="450px" height="100px" ></img>

+ a라는 태그에서 class속성 값이 lotto-bnt-current 인 곳에 현재 회차 데이터가 다른 데이터들과 함께 있다.

```
>import requests
>from bs4 import BeautifulSoup
>import pandas as pd
>
>
>url=requests.get("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90")
>html=BeautifulSoup(url.text)           #text로 번역
>html.select('a._lotto-btn-current')[0] #문자열이 아닌 html상태로 가져오기

<a class="_lotto-btn-current" href="#" nocr=""><em>972회</em>차 당첨번호 <span>2021.07.17</span></a>
```


##### >회차 정보만 가져오기

+ 그 중에서 <em> </em>에 해당하는 부분에 내가 가져오고 싶은 회차 데이터가 있음.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcBxtv6%2Fbtq9VjRAf9v%2FEIjhXrQkjXaaIkdZklp1IK%2Fimg.png" width="450px" height="100px" ></img>


```
>html.select('a._lotto-btn-current')[0].select('em')

[<em>972회</em>]

```

##### >회차의 숫자만 가져오기
```
>html.select('a._lotto-btn-current')[0].select('em')[0].text

'972회'
```

##### >'회'를 빈칸으로 만들기
```
>html.select('a._lotto-btn-current')[0].select('em')[0].text.replace('회','')

'972'
 ```
 
##### >current 라는 변수에 문자열로 되어있는 회차를 정수형으로 저장하기
 ```
 >current=int(html.select('a._lotto-btn-current')[0].select('em')[0].text.replace('회',''))
>current

972
```

##### >전체 코드
```
>import requests
>from bs4 import  BeautifulSoup
>import pandas as pd
>
>
>total=[]
>
>url=requests.get("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90")
>html=BeautifulSoup(url.text)
>current=int(html.select('a._lotto-btn-current')[0].select('em')[0].text.replace('회','')) #회를 빈칸으로 만들기
>
>for n in range(1,101):
>
>  url=requests.get(f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90+{n}%ED%9A%8C")
>  html=BeautifulSoup(url.text)
>
>  lotto_number=html.select('div.num_box')[0].select('span')
>  del lotto_number[6]
>
>  box=[]
>
>  for i in lotto_number:
>    box.append(int(i.text))
>
>  total.append(box)
>  
>  print('로또 {}회차 데이터 저장완료 : {}'.format(n,box))

로또 1회차 데이터 저장완료 : [10, 23, 29, 33, 37, 40, 16]
로또 2회차 데이터 저장완료 : [9, 13, 21, 25, 32, 42, 2]
로또 3회차 데이터 저장완료 : [11, 16, 19, 21, 27, 31, 30]
로또 4회차 데이터 저장완료 : [14, 27, 30, 31, 40, 42, 2]
로또 5회차 데이터 저장완료 : [16, 24, 29, 40, 41, 42, 3]
로또 6회차 데이터 저장완료 : [14, 15, 26, 27, 40, 42, 34]
..........생략.............
로또 94회차 데이터 저장완료 : [5, 32, 34, 40, 41, 45, 6]
로또 95회차 데이터 저장완료 : [8, 17, 27, 31, 34, 43, 14]
로또 96회차 데이터 저장완료 : [1, 3, 8, 21, 22, 31, 20]
로또 97회차 데이터 저장완료 : [6, 7, 14, 15, 20, 36, 3]
로또 98회차 데이터 저장완료 : [6, 9, 16, 23, 24, 32, 43]
로또 99회차 데이터 저장완료 : [1, 3, 10, 27, 29, 37, 11]
로또 100회차 데이터 저장완료 : [1, 7, 11, 23, 37, 42, 6]
```

