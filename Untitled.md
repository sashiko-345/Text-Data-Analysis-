```python
import os
import sys
import importlib,sys
importlib.reload(sys)
import jieba
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2,SelectKBest
import pickle

stop_words_path = "stopwords-zh.txt"

#获取停用词数据集
#可手动添加一些停用词
def read_stopwordset():
    with open(stop_words_path, "rb") as fp: 
        lines = fp.readlines()
        words = [' ','\r\n']
        for line in lines:
            words.append(line.strip())
    return words

#利用结巴分词器对文本内容进行分词
#输入要分词的句子
#获得分词结果，结果以列表形式返回
def fenci(sentence): 
    wordlist = jieba.lcut(sentence,cut_all=False,HMM=True) # 精确模式
    return wordlist

#将模型按照model_name进行保存
def save_model(model_name,model):
    feature_path = 'models/'+model_name+'.pkl'
    with open(feature_path, 'wb') as fw:
        pickle.dump(model, fw)

#将模型按照model_name进行读取
def load_model(model_name):
    tfidftransformer_path = 'models/'+model_name+'.pkl'
    return pickle.load(open(tfidftransformer_path, "rb"))
    
#获取停用词
stopwords = read_stopwordset()
#按类别获取文档内容
rawData = datasets.load_files("train",encoding = 'gbk',decode_error='ignore')
print("文件读取结束")

X_train = [] #文本内容
Y_train = [] #文本标签
count = [0,0,0,0,0,0,0,0,0] #控制训练集数量
for i in range(len(rawData.data)):
    if rawData.filenames[i][-3:]=='txt':
        if count[rawData.target[i]]<500:
            content = rawData.data[i]
            result = fenci(content)
            X_train.append(' '.join(result))
            Y_train.append(rawData.target[i])
            count[rawData.target[i]]+=1
        else:
            continue
print("文件预处理结束")


```

    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/fq/90_3kyc56x3gprlncwjfvbx00000gp/T/jieba.cache


    文件读取结束


    Loading model cost 1.356 seconds.
    Prefix dict has been built successfully.


    文件预处理结束



```python
#将文档内容转换为词-词频的模式
count_vec = CountVectorizer(min_df=0.1,stop_words=stopwords) #可以利用最大最小文档频率对词进行筛选max_df=0.6,min_df=0.2
X_train = count_vec.fit_transform(X_train)

save_model('count_vec',count_vec.vocabulary_)

#基于卡方的特征选择  ##文档频率法、信息增益法
select_chi2 = SelectKBest(chi2,k=100)
X_train = select_chi2.fit_transform(X_train,Y_train)
save_model('select_chi2',select_chi2)

#tfidf对词赋予权重
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
save_model('tfidf',tfidf)

print("文件特征选择结束")
```

    文件特征选择结束



```python
#利用SVM算法对文档内容进行分类学习 ##决策树、KNN算法 ###实践一下cart
clf = SVC()
clf.fit(X_train, Y_train)
save_model('svm',clf)

print("文件训练结束")

```

    文件训练结束



```python
rawData = datasets.load_files("test",encoding = 'gbk',decode_error='ignore')
X_test = []
Y_test = []
count = [0,0,0,0,0,0,0,0,0]
for i in range(len(rawData.data)):
    if rawData.filenames[i][-3:]=='txt':
        if count[rawData.target[i]]<100:
            content = rawData.data[i]
            result = fenci(content)
            X_test.append(' '.join(result))
            Y_test.append(rawData.target[i])
            count[rawData.target[i]]+=1
        else:
            continue
X_test = count_vec.transform(X_test)
X_test = select_chi2.transform(X_test)
test_tfidf = tfidf.transform(X_test)
Y_result = clf.predict(test_tfidf)

#计算准确率
correct = 0.0
for i in range(len(Y_result)):
    if Y_test[i] == Y_result[i]:
        correct+=1.0
print("在测试集中准确率为："+str(correct/len(Y_result)))

```

    在测试集中准确率为：0.7511111111111111



```python

```
