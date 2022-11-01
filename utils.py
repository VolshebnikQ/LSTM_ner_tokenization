import torch
from collections import Counter

#   Разделить данные (два столбика) на словарь и сборник тегов
def vocabulary_tags(text):
    vocabulary = set()
    tags = set()
    for line in text:
        if len(line) > 0:
            word, tag = line.split(' ')
            vocabulary.add(word)
            tags.add(tag)
            
    return vocabulary, tags

#   Подготовка данных для обработки нейронной сетью
def prepare_data(lines):
    text_w_tags = []
    text, tags = [],[]

    for line in lines:
        if len(line)>0:
            word, label = line.split(' ')
            text.append(word)
            tags.append(label)
        else:
            if len(text)>0:
                text_w_tags.append((text, tags))
            text, tags = [],[]

    return text_w_tags

#   Класс для токенизации словаря и тегов
class Converter():
    def __init__(self, vocabulary, tags):
        self.idx_to_word = sorted(vocabulary)
        self.idx_to_tag  = sorted(tags)

        self.word_to_idx = {word:idx for idx,word in enumerate(self.idx_to_word)}
        self.tag_to_idx  = {tag:idx for idx,tag in enumerate(self.idx_to_tag)}
        
    def words_to_index(self, words):
        return torch.tensor([self.word_to_idx[w] for w in words], dtype=torch.long)

    def tags_to_index(self, words):
        return torch.tensor([self.tag_to_idx[w] for w in words], dtype=torch.long)
    
    def indices_to_words(self, indices):
        return [self.idx_to_word[i] for i in indices]
    
    def indices_to_tags(self, indices):
        return [self.idx_to_tag[i] for i in indices]

#   Предсказать теги для подготовленных данных
def predict_tags(model, converter, text):

    encoded_text = converter.words_to_index(text)
    encoded_tags = model.predict_tags(encoded_text)
    decoded_tags = converter.indices_to_tags(encoded_tags)
    return decoded_tags

#   Получение статистики
def tag_statistics(model, converter, data):

    def tag_counter(predicted, ground):
        correct_tags = Counter()
        ground_tags  = Counter(ground)
        
        for tag_p, tag_g in zip(predicted, ground):
            if tag_p==tag_g:
                correct_tags[tag_g]+=1            
        return correct_tags, ground_tags
    
    
    total_correct, total_tags = Counter(), Counter()
    
    for text, tags in data:
        tags_pred              = predict_tags(model, converter, text)
        tags_correct, tags_num = tag_counter(tags_pred, tags)

        total_correct.update(tags_correct)
        total_tags.update(tags_num)

    return total_correct, total_tags