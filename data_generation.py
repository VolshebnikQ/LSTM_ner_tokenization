import csv
from random import randint
import pymorphy2
from numpy import NaN


label_main = ['ДРУГОЕ', 'ЧИСЛО', 'ЗНАК']    #   теги
label = ['B-ЧИСЛО','I-ЧИСЛО']    #   теги в формате BIO
in_l = ['','','','','']
theme = ['СУММА','РАЗНОСТЬ','ПРОИЗВЕДЕНИЕ','ЧАСТНОЕ']   #  Действия

morph = pymorphy2.MorphAnalyzer()

#   Типы задач на сложение или их количество
def getTypeText(type_sent, type_topic, num_x, num_y, isLen = False):
    if not isLen:
        x = num_x
        y = num_y
        text = ""
        labels = ""
        topic = ""
        answer = ""
        if(type_topic == 0):    #   СУММА
            topic = theme[type_topic]
            answer = x + y
            if(type_sent == 1):
                text = f'{x} + {y}'
                labels = '1 2 1'
            if(type_sent == 2):
                text = f'Выполнить сложении {x} и {y}.'
                labels = '0 2 1 0 1 0'
            if(type_sent == 3):
                text = f'Получить сумму {x} и {y}.'
                labels = '0 2 1 0 1 0'
            if(type_sent == 4):
                text = f'Посчитай {x} + {y}.'
                labels = '0 1 2 1 0'
            if(type_sent == 5):
                text = f'Вычисли сумму от чисел {x} и {y}.'
                labels = '0 2 0 0 1 0 1 0'
            if(type_sent == 6):
                text = f'Какое значение получится, если сложить {x} и {y}?'
                labels = '0 0 0 0 0 2 1 0 1 0'
            if type_sent == 7:
                text = f'Что будет, если взять сумму {x} и {y}?'
                labels = '0 0 0 0 0 2 1 0 1 0'
            if type_sent == 8:
                text = f'А сумма {x} и {y}?'
                labels = '0 2 1 0 1 0'
        if(type_topic == 1):    #   ВЫЧИТАНИЕ
            topic = theme[type_topic]
            answer = x - y
            if(type_sent == 1):
                text = f'{x} - {y}'
                labels = '1 2 1'
            if(type_sent == 2):
                text = f'Вычтите {x} из {y}.'
                labels = '2 1 0 1 0'
            if(type_sent == 3):
                text = f'Найдите разность от чисел {x} и {y}.'
                labels = '0 2 0 0 1 0 1 0'
            if(type_sent == 4):
                text = f'Разность числа {x} и числа {y}'
                labels = '2 0 1 0 0 1'
            if(type_sent == 5):
                text = f'Чему равна разность чисел {x} и {y}?'
                labels = '0 0 2 0 1 0 1 0'
            if(type_sent == 6):
                text = f'Посчитай {x} - {y}'
                labels = '0 1 2 1'
            if(type_sent == 7):
                text = f'Вычислить разность {x} и {y}'
                labels = '0 2 1 0 1'
            if(type_sent == 8):
                text = f'Какой стоит ожидать результат, если вычесть из {x} число {y}?'
                labels = '0 0 0 0 0 0 2 0 1 0 1 0'
        if(type_topic == 2):    #   ПРОИЗВЕДЕНИЕ
            topic = theme[type_topic]
            answer = x * y
            if(type_sent == 1):
                text = f'{x} * {y}'
                labels = '1 2 1'
            if(type_sent == 2):
                text = f'Посчитай {x} * {y}'
                labels = '0 1 2 1'
            if(type_sent == 3):
                text = f'Что будет умножив {x} на {y}?'
                labels = '0 0 2 1 0 1 0'
            if(type_sent == 4):
                text = f'Чему равно произведение {x} и {y}?'
                labels = '0 0 2 1 0 1 0'
            if(type_sent == 5):
                text = f'И найдите произведение {x} и {y}.'
                labels = '0 0 2 1 0 1 0'
            if(type_sent == 6):
                text = f'Вычислить произведение чисел {x} и {y}'
                labels = '0 2 0 1 0 1'
            if(type_sent == 7):
                text = f'Какой стоит ожидать результат, если умножить {x} на {y}?'
                labels = '0 0 0 0 0 0 2 1 0 1 0'
            if(type_sent == 8):
                text = f'Произведение от числа {x} и числа {y}'
                labels = '2 0 0 1 0 0 1'
        if(type_topic == 3):    #   ЧАСТНОЕ
            topic = theme[type_topic]
            answer = x / y
            if(type_sent == 1):
                text = f'{x} / {y}'
                labels = '1 2 1'
            if(type_sent == 2):
                text = f'А посчитай {x} / {y}'
                labels = '0 0 1 2 1'
            if(type_sent == 3):
                text = f'А что будет, если поделить {x} на {y}?'
                labels = '0 0 0 0 0 2 1 0 1 0'
            if(type_sent == 4):
                text = f'Чему равно частное {x} и {y}?'
                labels = '0 0 2 1 0 1 0'
            if(type_sent == 5):
                text = f'Найдите частное {x} и {y}.'
                labels = '0 2 1 0 1 0'
            if(type_sent == 6):
                text = f'Вычислить частное {x} и {y}'
                labels = '0 2 1 0 1'
            if(type_sent == 7):
                text = f'Какой результат стоит ожидать, если поделить {x} на {y}?'
                labels = '0 0 0 0 0 0 2 1 0 1 0'
            if(type_sent == 8):
                text = f'Частное чисел {x} и {y}'
                labels = '2 0 1 0 1'
        return text.lower(), labels, topic, answer
    else:
        return 8,3

#   Найти посторонний символы и выделить их пробелом
def findOther(string):
    other = "<>.,?:=({[]})&^$#@!~`_№'';"
    for o in other:
        i = -2
        while string.find(o,i+2) != -1:
            i = string.find(o,i+2)
            try:
                if string[i-1] != " ":
                    string = string[:i] + " " + string[i:]
                    i += 1
                if string[i+1] != " ":
                    string = string[:i+1] + " " + string[i+1:]
            except Exception:
                continue
    return string

#   Записать массив словаря
with open("data/dictionary.csv", mode="w", encoding='utf-8') as dictionary_file:
    type_sent_count, type_topic_count = getTypeText(NaN,NaN,NaN,NaN,True)
    for type_topic in range(type_topic_count+1):
        for type_sent in range(1, type_sent_count + 1):
            for i in range(1000):
                x = randint(1, 1000)
                y = randint(1, 1000)
                string = getTypeText(type_sent, type_topic, x, y, False)
                text = findOther(string[0]).strip().split(' ')
                labels = string[1].split(' ')
                for el in range(len(text)):
                    if label_main[int(labels[el])] == "ЧИСЛО":
                        start = "NUM"
                        line = start + " " + label[0] + '\n'
                        dictionary_file.write(line) 
                        for char in text[el]:
                            line = char + " " + label[1] + '\n'
                            dictionary_file.write(line)
                    elif label_main[int(labels[el])] == "ЗНАК":
                        morph_word = morph.parse(text[el])[0]
                        line = morph_word.normal_form  + " " + string[2] + '\n'
                    else:
                        morph_word = morph.parse(text[el])[0]
                        line = morph_word.normal_form  + " " + label_main[int(labels[el])] + '\n'
                    dictionary_file.write(line)
                dictionary_file.write('\n')

#   Узнать размер словаря
with open("data/dictionary.csv", encoding='utf-8') as file:
    lines = file.read().strip().split('\n')
    print(len(lines))