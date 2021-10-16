# Material Recognition 

Texture, Material recognition를 연구하며 제안한 custom network를 정리하는 공간입니다.

- Multi-view 환경에서 surface reflectance를 encoding하기 위한 네트워크 구조
- Backbone에서 추출된 feature 들 간의 view correlation을 위해 LSTM을 사용
- 논문 작성 후, 추가 내용 업로드 예정...TBD...

## Basic Cumtom Network
- oneStream_multiView_Net : Color multi view 이미지들을 LSTM을 이용하여 surface reflectance를 encoding하여 재질을 분류하는 구조
- twoStream_multiView_Net : Color multi view 이미지들로 differential image를 생성하고, backbone net을 2개 사용하여 각각 color, differential image를 input으로 받고 output feature를 concat하여 LSTM에 주입하는 구조

## Main Network
TBD.. 
