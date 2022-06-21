from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
import matplotlib.pyplot as plt
# 데이터 샘플 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i -3, '?') for i in x_test[0]])

def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results
# 데이터 정규화
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)
# 원 핫 인코딩
one_hot_train_labels = to_categorical(y_train)
one_hot_test_labels = to_categorical(y_test)
# 모델 생성
model = models.Sequential()
# 히든레이어 생성
# 입력 노드는 출력노드보다 커야함
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
#출력층, 분류 개수가 46개므로 노드가 46개가 된다.
model.add(layers.Dense(46, activation='softmax'))
# 다중 분류이므로 loss = categorical_crossentropy, optimizer 는 rmsprop
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
# 9번째 세대에서 과대적합 발생
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

# 모델 평가
results = model.evaluate(x_test, one_hot_test_labels)

loss = history.history['loss']
val_loss = history.history['val_loss']
# 사용한 에포크 만큼 그래프에 그리기
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='training Loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title("Train and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
