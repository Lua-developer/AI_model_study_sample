# AI_model_study_sample

## release
2022-06-21 Reuter 데이터셋 모델 훈련 및 검증, 샘플데이터를 이용한 데이터 예측  
2022-07-01 MNIST CNN 이미지 분류 모델 학습  
MNIST 설명 : https://blog.naver.com/bjjy1113/222796316586  
2022-07-02 VGG16 모델을 이용한 CNN 전이학습 (Kaggle dogs and Cats Dataset 사용)  
2022-08-05 Kaggle 폐렴 데이터셋을 이용한 CNN 이진 분류 모델 학습  
# 2022-08-05 모델 review
총 학습 시간 : 53분 48초  
학습데이터 : 5216개 테스트 데이터 : 624개 약 9:1  
batch size = 32, epoch = 25  
학습 데이터 출처 : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
모델 레퍼런스 : https://www.kaggle.com/code/khoongweihao/covid-19-ct-scan-xray-cnn-detector  

최종 학습 결과(epoch 25) : train_accuracy = 96%, val_accuracy = 74%  
![image](https://user-images.githubusercontent.com/83262616/183012493-081f965e-bb3e-4c03-a639-9eb8a0c0d953.png)  
시각화  
![image](https://user-images.githubusercontent.com/83262616/183048198-32dcd1da-a2ca-4f7f-a1d7-a103f3a7fed0.png)
이미지 데이터셋 전처리 과정 : keras.preprocessing.image의 ImageDataGenerator 사용  

1차 학습 리뷰  
훈련 성능과 손실의 경우 지속적으로 증가, 감소세를 보임(훈련성능은 높은 성능을 보였음)  
검증 성능과 손실의 경우 상승과 하강을 반복, 모델 개선이 필요  
모델의 크기에 비해 많은 epoch을 사용하여 시간의 소모가 큼 elary stopping을 이용하여 높은 성능을 보일때 stop 하는 것을 고려

## 1차 학습에서 배운 점  
함수형 API를 이용한 CNN 신경망 생성 및 구현에 대해 이해할 수 있었음. 하지만 신경망은 기존 논문이나 레퍼런스가 많으므로 처음부터 쌓기 보단 레퍼런스를 참조하거나 기존 모델의 전이학습을 통해 재활용 하는 것이 시간 단축 및 정확도 향상에 기여 가능함.  
2차 학습 전 기존 테스트 셋을 이용하여 다시 학습 결과 기존의 학습 결과에 영향을 받음, 테스트데이터의 증강 및 shuffle 및 수정이 필요  
케라스 라이브러리를 사용하여 이미지 제네레이팅이 아닌, opencv를 이용하여 데이터셋에 대해 정규화 및 증강을 시도 해 볼 필요가 있다 느낌.  
모델에서 입력 받을때 150 * 150 * 3, 컬러 이미지를 입력 받고 있는데 현재 폐의 Xray 이미지는 흑백 이미지므로 이미지에 대해 GrayScale로 변환 후 전처리 하여 입력 시 더 좋은 성능을 기대 가능 할것 같다.
