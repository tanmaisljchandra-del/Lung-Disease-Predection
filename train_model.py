import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET = "dataset"

labels = ["Healthy","Pneumonia","Bronchitis","Asthma","COPD","URTI"]

X = []
y = []

for i,label in enumerate(labels):

    folder = os.path.join(DATASET,label)

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        audio, sr = librosa.load(path,duration=5)

        mel = librosa.feature.melspectrogram(y=audio,sr=sr)

        mel = librosa.power_to_db(mel)

        mel = mel[:128,:128]

        X.append(mel)
        y.append(i)

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0],128,128,1)

y = to_categorical(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(6,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=10)

model.save("model.h5")

print("Model trained successfully")