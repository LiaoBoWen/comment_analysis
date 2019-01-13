import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

# 分词和去掉停用词
def processing_word(x,stop_words):
    words_stop = ['QQ炫舞','qq炫舞','Qq炫舞','炫舞','劲舞团','劲舞','端游','手游','炫舞时代','棒棒哒','腾讯','网易']
    for word_stop in words_stop:
        jieba.suggest_freq(word_stop,True)

    cut_word = jieba.cut(x.strip())
    word = [word for word in cut_word not in stop_words]
    return word

def get_stop_words():
    stop_words_list = []
    with open('stop_word.txt','r') as stop_words_file:
        print('文件：',stop_words_file)
        for line in stop_words_file:    #还能有这种操作？！
            stop_words_list.append(line.strip())
    return stop_words_list

def load_file_and_split():
    positive_comment = pd.read_excel('comment.xls',header=None,heet_name='positive_comment')
    negative_comment = pd.read_excel('comment.xls',header=None,sheet_name='negative_comment')
    stop_words = get_stop_words()
    print(positive_comment['words'])
    positive_comment['words'] = positive_comment[0].apply(processing_word,args=(stop_words,))
    negative_comment['words'] = negative_comment[0].apply(processing_word,args=(stop_words,))
    x = np.concatenate((positive_comment['words'],negative_comment['words']))
    y = np.concatenate((np.ones(len(positive_comment)),np.zeros(len(negative_comment))))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
    return x,x_train,x_test,y_train,y_test