from keras import models, layers
# mnist dataset
from keras.datasets import mnist
# one hot encording
from tensorflow.keras.utils import to_categorical
# model setup
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# 훈련 집합과 검증집합 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 훈련 데이터, 검증데이터 정규화
x_train = x_train.reshape((60000, 28,28,1))
x_train = x_train.astype('float32')/ 255
x_test = x_test.reshape((10000,28,28,1))
x_test = x_test.astype('float32')/ 255
# 원 핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 모델 컴파일
model.compile(loss='categorical_crossentropy', metrics=['acc'])
# 모델 학습
history = model.fit(x_train, y_train, epochs=5, batch_size = 64)
# 요약
model.summary()
