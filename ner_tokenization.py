import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os.path
import random

import pymorphy2

import utils


torch.manual_seed(1)
morph = pymorphy2.MorphAnalyzer()

theme = ['СУММА','РАЗНОСТЬ','ПРОИЗВЕДЕНИЕ','ЧАСТНОЕ']   #  Действия

#   Подготовить текст для прочтение его моделью
def prepare_input_text(text, norm = False):
    other = "<>.,?:=({[]})&^%$#@!~`_№';1234567890"
    end = False
    i = 0
    string = ""
    for s in text.split(' '):
        if norm:                                #   Нормализуем входной текст
            morph_word = morph.parse(s)[0]
            s = morph_word.normal_form
        string += " " + s
    string = string.strip()
    while not end:                              #   Ищим цифры и ставим перед ними тэг "NUM"
        prev_digit = False
        try:
            int(string[i])
            if not prev_digit:
                space = " "
                if string[i-1] == " ":
                    space = ""
                string = string[:i] + space + "NUM " + string[i:]
                i += 5
            prev_digit = True
        except ValueError:
            prev_digit = False
        except IndexError:
            end = True
        i += 1
    for o in other:                             #   Обособляем пробелом цифры и пунктуационные знаки
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
    if norm:
        print(f'Нормальная форма -> {string}')
    return string.split(' ')

#   Собрать из вывода модели числа по разрядам ['NUM', '1', '2', 'NUM', '5', '5'] -> ['12', '55']
def get_nums(nums_words):
    nums = []
    n = ""
    for i in range(1, len(nums_words)+1):
        try:
            if nums_words[i] == "NUM":
                nums.append(n)
                n = ""
            else:
                n += nums_words[i]
        except Exception:
            nums.append(n)

    return nums

#   Создать класс нейронной сети LSTM
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):

        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words):
        embeds = self.word_embeddings(words)
        lstm_out, _ = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

    def predict_tags(self, words):
        with torch.no_grad():
            tags_pred = model(words).numpy()
            tags_pred = np.argmax(tags_pred, axis=1)
            
        return tags_pred

#   Собрать модель или обучить
def getModel(path_model, datafile, is_train):
    with open(datafile, encoding='utf-8') as r_file:
        lines = r_file.read().strip().split('\n')

    vocabulary, tags = utils.vocabulary_tags(lines)         #   Получем список слов и тегов из таблицы

    words_to_tags = sorted(utils.prepare_data(lines), key=lambda A: random.random())    #   Перемешать исходные данные
    converter = utils.Converter(vocabulary, tags)                                       #   Создаём словари для слов и тегов

    print(len(words_to_tags))                   # Считаем количество данных
    len_data = len(words_to_tags)
    train = int(len(words_to_tags)*0.9)
    training_data = words_to_tags[     :train]
    test_data = words_to_tags[len_data-train:]

    EMBEDDING_DIM = 32                          #   Длина выходных значений для слоя EMBEDDING
    HIDDEN_DIM = 32                             #   Длина выходных значений скрытого слоя
    VOCAB_SIZE = len(converter.word_to_idx)     #   Длина One Hot Encoding для 
    TAGSET_SIZE = len(converter.tag_to_idx)     #   Длина One Hot Encoding для тегов

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE)
    loss_function = nn.NLLLoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if is_train:
        for epoch in range(1):
            for i, (text, tags) in enumerate(training_data):
                
                model.zero_grad()
                
                encoded_text = converter.words_to_index(text)
                encoded_tags = converter.tags_to_index(tags)
                
                tag_scores = model(encoded_text)

                loss = loss_function(tag_scores, encoded_tags)
                loss.backward()
                optimizer.step()

                if i % 500 == 0:    #   Шаг модели
                    print(loss)
                    
                if i == 3000:       
                    break
        torch.save(model.state_dict(), path_model)
    else:
        if os.path.exists(path_model):
            model.load_state_dict(torch.load(path_model))
            model.eval()
        else:
            pass

    return model, converter, test_data

#   Предсказать теги для не подготовленного текста
def predict_tags(model, converter, text, is_raw = False):
    prep_text = text
    if type(text) != list:
        prep_text = prepare_input_text(text, True)
    
    encoded_text = converter.words_to_index(prep_text)
    encoded_tags = model.predict_tags(encoded_text)
    decoded_tags = converter.indices_to_tags(encoded_tags)

    if is_raw:
        return decoded_tags

    nums = []
    sign = []
    start_num = False
    print()
    print('Предсказанные теги')
    for i in range(len(decoded_tags)):
        print(prep_text[i] + " --- " + decoded_tags[i])
        if decoded_tags[i] == "B-ЧИСЛО":
            start_num = True
            nums.append(prep_text[i])

        if start_num and decoded_tags[i] == "I-ЧИСЛО":
            nums.append(prep_text[i])

        if decoded_tags[i] != "I-ЧИСЛО" and decoded_tags[i] != "B-ЧИСЛО":
            start_num = False
        
        if theme.count(decoded_tags[i]) > 0:
            sign.append(decoded_tags[i])

    return decoded_tags, prepare_input_text(text), [get_nums(nums), sign]


model, converter, test_data = getModel('models/ner_tokenization.pt', 'data/dictionary.csv', False)

#   Предсказать теги для нового текста.
print()
text = "Что если найти разность чисел 55 и 61"
print(f'На вход идёт')
print(f'-> {text}')
print()
tags_pred, end_text, arr = predict_tags(model, converter, text)

print()
print(f'Числа {arr[0]}, знак {arr[1]}')
print()

#   Оформление проверки адекватности модели
#total_correct, total_tags = utils.tag_statistics(model, converter, test_data)

#print('Статистика верно предсказанных тэгов:\n')
#for tag in total_tags.keys():
#    print('для {}:'.format(tag))
#    print('  корректно:\t', total_correct[tag])
#    print('      всего:\t',   total_tags[tag])
#    print('% корректно:\t', 100 * (total_correct[tag] / float(total_tags[tag])))
#    print()
#
#print('----------')
#print('в итоге:')
#print('  корректно:\t', sum(total_correct.values()))
#print('      всего:\t', sum(total_tags.values()))
#print('% корректно:\t', 100 * (sum(total_correct.values()) / sum(total_tags.values())))



