# 2023-MiraAsset

[공모전] LLM을 활용한 금융서비스 제안 : 주식 테마 네트워크 시각화

## 1. 대회 정보
- 주관 : 미래에셋증권 X NAVER Cloud 공동주관
- 참가 대상 : 대학(원)생으로 이루어진 팀
- 공모 주제 : 생성형 AI시대, AI와 빅데이터로 내가 만드는 금융서비스
  - 생성형 AI 부문 : 초거대 언어모델을 활용한 대고객 금융 서비스 제안
  - 빅데이터 부문 : 고객 및 시장 데이터를 활용한 대고객 금융 서비스 제안
- **최종 결과 : 대상 수상**

</br>

## 2. 대회 기간
- 예선: 2023/07/01 ~ 2023/08/02
- 본선: 2023/08/09 ~ 2023//09/13
- 결선 및 멘토링: 2023/09/19 ~ 2023/10/10

</br>

## 3. 프로젝트 개요

- 배경
  1. 모멘텀의 부재에 따라 뚜렷한 주도주 없이 테마주 순환 장세 지속
  2. 변동성이 큰 테마주 투자에는 손실의 위험성 존재

- 목표 : 신속하게 새로운 테마를 탐지하고, 테마간의 관계를 시각화하여 제공

</br>

## 4. 프로젝트 내용

![image](https://github.com/user-attachments/assets/fa5533be-be6d-4698-9763-d68239bfbec7)

### 주식 테마 추출
- 사용 모델 : GPT 3.5 Turbo
- 뉴스 기사로부터 연관된 주식 테마 추출

</br>

### 테마주 선정
- 사용 모델 : KR-SBERT (한국어 문장 사전학습 모델)
- 방법
  1. 각 기업별 공시자료, IR 자료 문서와 테마-연관어 문서 임베딩
  2. 기업, 테마 임베딩 벡터 간의 코사인 유사도 계산
  3. 유사도 0.4 이상인 상위 20개 기업 추출

</br>

### 테마 네트워크 구축
- 방법
  1. 테마 유사도 행렬을 기반으로 네트워크 구축
  2. Node2Vec + Kmeans 방법론을 통해 테마 군집화
- 프로토타입 구현 결과
![테마빛나비1](https://github.com/suinkim19/themebitnavi/assets/103398227/c44571ed-c2f9-4cc1-a2c5-5c2e78e51275)
![테마빛나비2](https://github.com/suinkim19/themebitnavi/assets/103398227/3a29e908-510d-4823-8f85-9380fd63c15c)

</br>

### 서비스 제안
1. 다각화된 테마 정보 제공
2. 개인화된 테마 추천에 활용 가능
3. 고객의 관심 테마 정보를 통한 다이렉트 인덱싱에 활용 가능
   
![image](https://github.com/user-attachments/assets/6feed8d6-e6d5-4f41-baf2-6b3a462b8682)


</br>

## 5. 코드 정리 

ThemeGPT.py : ThemeGPT를 통한 테마 추출, 테마명 통합, 테마명 요약에 대한 함수 파일

theme_index.py : 테마지수 계산에 사용되는 함수 파일

theme_assess.py : ThemeGPT 성능 평가에 사용된 함수 파일

network_function.py : 테마 네트워크 구축 및 시각화에 사용된 함수 파일

clova_summary.py : CLOVA summary API 사용에 사용된 함수 파일

article_processing.py : 뉴스 데이터의 전처리에 사용된 함수 파일

ThemeGPT.ipynb : 테마 추출, 테마명 통합, 테마명 요약 등 GPT API를 사용하는 모든 모델에 대한 예시

theme_stock_and_index.ipynb : GPT가 추출한 각 테마에 속한 종목들을 선정하고, 테마지수를 계산하는 예시

theme_network.ipynb : 테마 네트워크를 구축하고 시각화하는 코드의 예시

</br>

### API 발급 리스트
- DART Open API 키 발급하기
  1. https://opendart.fss.or.kr/ 접속 
  2. 인증키 신청/관리 -> 인증키 신청
  3. 계정 생성 및 API키 발급
  4. 공시자료_crawl.ipynb에서 API 키를 입력
 
- GPT API 키 발급하기
  1. https://platform.openai.com/ 접속
  2. 로그인 후 API로 이동
  3. 오른쪽 상단 Personal 클릭 후 View API key에 접속하여 API key 발급
  4. ThemeGPT.ipynb에서 API 키를 입력
    
- CLOVA SUMMARY API 키 발급하기
  1. https://console.ncloud.com/dashboard에 로그인 후 접속
  2. 콘솔에서 Services > AI, NAVER API > Application에서 Application 등록
  3. CLOVA Summary API를 발급하고, 인증 정보에서 Client ID 및 Password 확인
  4. clovasummary.py에서 ID, password를 입력

