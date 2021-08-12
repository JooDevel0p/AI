# 데이터 크롤링 : 웹페이지를 그대로 가져와 데이터를 추출하는 것

 

* HTML : 사이트 구조와 내용  

* JSP : 서버연동, 데이터 저장, 다양한 기능

* CSS : 포토샵

 

 **데이터 크롤링**을 위해서는 내용을 가져오기 때문에 **HTML** 지식만으로도 충분하다.


--------

## 로또 데이터 크롤링


##### >크롤링할 사이트의 데이터 저장하기

```
import requests     
from bs4 import BeautifulSoup  
import pandas as pd  

url=requests.get("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90+777%ED%9A%8C")
url
```

+ requests는 URL주소에 있는 내용을 요청할 때 사용하는 모듈
+ BeautifulSoup는 python언어로 HTML을 다루는 라이브러리
+ url 로 불러온 데이터는 가공되지 않은 데이터로 정리되어 있지 않다. 또한 문자열 형태로 저장되기때문에 html 형태로 변환 시켜줘야 한다.



##### >문자열 데이터를 html로 반환하기

```
>html=BeautifulSoup(url.text)
>html
```

##### >가져오고 싶은 데이터 부분 추출하기

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEeUmE%2Fbtq9DUwyhIo%2FiU5etu6R4XkPikNwl49Hk0%2Fimg.png" width="500px" height="120px" alt="가져오고 싶은 데이터 부분"></img>

```
>html.select('div.num_box')

[<div class="num_box"> <span class="num ball6">6</span> <span class="num ball12">12</span> <span class="num ball17">17</span> <span class="num ball21">21</span> <span class="num ball34">34</span> <span class="num ball37">37</span> <span class="bonus">보너스번호</span> <span class="num ball18">18</span> <a class="btn_num" href="https://www.dhlottery.co.kr/gameResult.do?method=myWin" nocr="" onclick="return goOtherCR(this, 'a=nco_x5e*1.contents&amp;r=1&amp;i=0011AD9E_0000009BBC09&amp;u=' + urlencode(this.href));" target="_blank">내 번호 당첨조회</a> </div>]
```
+ 태그이름 : div ,클래스의 속성값 이름 : num_box
+ 리스트로 불러왔기 때문에 html 꺼내야 한다.

```
>html.select('div.num_box')[0]
<div class="num_box"> <span class="num ball6">6</span> <span class="num ball12">12</span> <span class="num ball17">17</span> <span class="num ball21">21</span> <span class="num ball34">34</span> <span class="num ball37">37</span> <span class="bonus">보너스번호</span> <span class="num ball18">18</span> <a class="btn_num" href="https://www.dhlottery.co.kr/gameResult.do?method=myWin" nocr="" onclick="return goOtherCR(this, 'a=nco_x5e*1.contents&amp;r=1&amp;i=0011AD9E_0000009BBC09&amp;u=' + urlencode(this.href));" target="_blank">내 번호 당첨조회</a> </div>
```

##### >각 번호 데이터 추출하기

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc6yhmX%2Fbtq9C98OCTv%2FT9JB1Uuqxw2ch1hXe0eIsK%2Fimg.png" width="500px" height="120px" alt="가져오고 싶은 데이터 부분"></img>

```
>lotto_number=html.select('div.num_box')[0].select('span')
>lotto_number

[<span class="num ball6">6</span>,
 <span class="num ball12">12</span>,
 <span class="num ball17">17</span>,
 <span class="num ball21">21</span>,
 <span class="num ball34">34</span>,
 <span class="num ball37">37</span>,
 <span class="bonus">보너스번호</span>,
 <span class="num ball18">18</span>]
 ```
+ 각 번호의 태그 이름: span


##### >보너스 번호 삭제하기

```
>lotto_number=html.select('div.num_box')[0].select('span')
>del lotto_number[6]
>lotto_number

[<span class="num ball6">6</span>,
 <span class="num ball12">12</span>,
 <span class="num ball17">17</span>,
 <span class="num ball21">21</span>,
 <span class="num ball34">34</span>,
 <span class="num ball37">37</span>,
 <span class="num ball18">18</span>]
 ```
 
 
 ##### >텍스트만 불러오기
 
 ```
 >for i in lotto_number:
>    print(i.text)
6
12
17
21
34
37
18
```
+ 여기서 텍스트란, 로또 숫자들 (6,12,17,21,34,37,18)


##### >box라는 리스트 생성하여 로또 번호들 저장

```
>box=[]
>for i in lotto_number:
>   box.append(i.text)
>box

['6', '12', '17', '21', '34', '37', '18']
```


##### >box라는 리스트 생성하여 로또 번호들 저장

```
import requests
from bs4 import BeautifulSoup
import pandas as pd

total=[]

for n in range(1,101):

    url=requests.get(f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90+{n}%ED%9A%8C")
    html=BeautifulSoup(url.text)

    lotto_number=html.select('div.num_box')[0].select('span')
    del lotto_number[6]

    box=[]

    for i in lotto_number:
      box.append(int(i.text))
```
+ for n in range(1,101): 로또 1회부터 100회 까지
+ 주소 중 '777' 은 777회를 뜻하며 n으로 바꿔주어 1회부터 100회까지 조회 가능 하게 해야함.
+ 주소 앞에 f 붙이기


##### >로또 1회부터 100회 까지 로또 번호를 불러오는 전체 코드 및 실행

```
>import requests
>from bs4 import BeautifulSoup
>import pandas as pd
>
>total=[]
>
>for n in range(1,101):
>
>    url=requests.get(f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90+{n}%ED%9A%8C")
>    html=BeautifulSoup(url.text)
>
>    lotto_number=html.select('div.num_box')[0].select('span')
>    del lotto_number[6]
>
>    box=[]
>
>    for i in lotto_number:
>      box.append(int(i.text))
>
>    total.append(box)
>
>    print('로또 {}회차 데이터 저장완료 : {}'.format(n,box))


로또 1회차 데이터 저장완료 : [10, 23, 29, 33, 37, 40, 16]
로또 2회차 데이터 저장완료 : [9, 13, 21, 25, 32, 42, 2]
로또 3회차 데이터 저장완료 : [11, 16, 19, 21, 27, 31, 30]
로또 4회차 데이터 저장완료 : [14, 27, 30, 31, 40, 42, 2]
로또 5회차 데이터 저장완료 : [16, 24, 29, 40, 41, 42, 3]
로또 6회차 데이터 저장완료 : [14, 15, 26, 27, 40, 42, 34]
로또 7회차 데이터 저장완료 : [2, 9, 16, 25, 26, 40, 42]
....생략.....
로또 97회차 데이터 저장완료 : [6, 7, 14, 15, 20, 36, 3]
로또 98회차 데이터 저장완료 : [6, 9, 16, 23, 24, 32, 43]
로또 99회차 데이터 저장완료 : [1, 3, 10, 27, 29, 37, 11]
로또 100회차 데이터 저장완료 : [1, 7, 11, 23, 37, 42, 6]
```



##### >total에 저장된 모든 데이터 불러오기
```
>total


[[10, 23, 29, 33, 37, 40, 16],
 [9, 13, 21, 25, 32, 42, 2],
 [11, 16, 19, 21, 27, 31, 30],
 [14, 27, 30, 31, 40, 42, 2],
 [16, 24, 29, 40, 41, 42, 3],
 ...생략....
 [1, 3, 8, 21, 22, 31, 20],
 [6, 7, 14, 15, 20, 36, 3],
 [6, 9, 16, 23, 24, 32, 43],
 [1, 3, 10, 27, 29, 37, 11],
 [1, 7, 11, 23, 37, 42, 6]]
 ```
 
##### >데이터를 표,엑셀로 나타내기

```
>df=pd.DataFrame(total, columns=['번호1','번호2','번호3','번호4','번호5','번호6','보너스번호'])
>df
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcfj74C%2Fbtq9EhFkB1w%2FQcDy3HQMwPmUlSttQ96eGK%2Fimg.png" width="350px" height="350px" ></img>
+ pandas에서는 표를 DataFrame 이라 부른다.


##### >df변수에 저장되어있는 total 데이터를 엑셀 파일에 저장하기

```
df.to_excel('lotto.xlsx')
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc6cdcF%2Fbtq9yMfLJU1%2F13VELoVRtk3ptaNbLyqOx0%2Fimg.png" width="300px" height="200px" ></img>

