'''
Data set link : https://www.kaggle.com/c/dogs-vs-cats
VGG16 모델을 이용한 전이학습 예제
'''
# keras.applications 에서 사전 학습된 모델을 제공함
from keras.applications.vgg16 import VGG16
import os
import numpy as np
# 이미지 전처리 함수
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
# 특성 추출을 위한 함수
def extract_features(diretory, sample_count) :
    features = np.zeros(shape=(sample_count, 4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        diretory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for input_batch, label_batch in generator :
        features_batch = conv_base.predict(input_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i+1) * batch_size] = label_batch
        i += 1
        if i * batch_size >= sample_count :
            break
    return features, labels
# VGG16 모델 사용
# include_top을 False로 파라메터를 주면 최상단 출력층을 사용하지 않음, (커스텀)
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# 훈련, 검증, 테스트 데이터셋의 경로 설정
base_dir = './datasets/cats_and_dogs_small'
# os.path.join 함수는 기존 경로와 임의의 경로를 합쳐준다.
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# 적은 데이터를 사용하므로 학습 전 데이터 증식
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
# 특성 추출
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
# Dense층에 주입하기 위하여 사이즈 재조정
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000,4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4*4*512))
# 출력층 재정의
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=(4*4*512)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
# 이진 분류 이므로 loss 함수로 binary_crossentropy 함수 사용
model.compile(loss='binary_crossentropy', metrics=['acc'])
# 훈련
history = model.fit(train_features, train_labels, batch_size=20,
                    epochs=30,
                    validation_data=(validation_features, validation_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# 훈련, 검증 정확도 분석
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
# 훈련, 검증 로스율 분석
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('training and validation loss')
plt.legend()

plt.show()

model.summary()
