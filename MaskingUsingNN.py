import random
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
import pymorphy2
import re
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras



def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name)

        print("Подключение к БД Mysql DB успешно установлено")
    except Error as e:
        print(f"Ошибка '{e}' Обнаружена")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Запрос успешно осуществлен")
    except Error as e:
        print(f"Ошибка '{e}' обнаружена")


def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"Ошибкаr '{e}' обнаружена")

connection = create_connection("localhost", "root", "password", "testmodel")
select_persdata = "SELECT * from userdata"
numbers = execute_read_query(connection, select_persdata)
select_persdata2 = "SELECT * from test"
numbers2 = execute_read_query(connection, select_persdata2)

for i in numbers:
    print(i)

arr = np.asarray(numbers)
arr2 = np.asarray(numbers2)
dta = pd.DataFrame(arr)
dta2 = pd.DataFrame(arr2)

print(arr)
print("///////////////////////////////////")
print(dta)

categories = {}
for key, value in enumerate(dta[2].unique()):
    categories[value] = key + 1

# Запишем в новую колонку числовое обозначение категории
dta[3] = dta[2].map(categories)

total_categories = len(dta[2].unique())

dataforlearn = dta[1]
labelsfortext = dta[3]
datafortest = dta2[1]

tokenizer1 = Tokenizer()
tokenizer2 = Tokenizer()
tokenizer1.fit_on_texts(dataforlearn.tolist())
textSequences = tokenizer1.texts_to_sequences(dataforlearn.tolist())
tokenizer2.fit_on_texts(datafortest.tolist())
Testtext=tokenizer2.texts_to_sequences((datafortest.tolist()))

total_words = len(tokenizer1.word_index)
print('В словаре {} слов'.format(total_words))

X_train, y_train = textSequences, labelsfortext
X_test = Testtext


# словарь
num_words = 781

print(u'Преобразуем данные в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
y_train = tf.keras.utils.to_categorical(y_train, num_classes=None)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=400)

model.save("model1")

test_loss, test_acc = model.evaluate(X_train, y_train)
print(test_acc)

predictions = model.predict(X_test)

print('Введите номер строки: ')
n=int(input())

print(predictions[n])

print(np.argmax(predictions[n]))