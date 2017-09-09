import re  # 匹配正则表达式，用于查找
import numpy as np
from gensim.models.doc2vec import Doc2Vec # 用于进行doc2vec学习，将文章段落转化为向量
import math
import jieba


# 用于诊断部分 目前使用的是基于西医的990种疾病
class Disease:
    def __init__(self, input_file, model, num):
        self.inp = input_file
        self.model = model
        self.num = num
    # 从文件中按行读取数据并且去除空行，存储为一个已分词的list，返回该list
    def get_input(self):
        data = []
        stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
        stopwords = re.split(r'\n', stopwords)
        for i in open(self.inp, 'r', encoding='utf-8').read():
            if i != '\n' and i not in stopwords:
                data.append(i)
        return data
    def sen2vec(self, model, sentence):
        model.random.seed(0)
        vec = model.infer_vector(sentence)
        return vec
    def sim(self, vec, num):
        lst_arr = np.load('diseases.npy')
        inp_arr = np.array(vec)
        lst = []
        for i in range(len(lst_arr)):
            dis = np.linalg.norm(lst_arr[i] - inp_arr)
            lst.append((dis, i))
        lst.sort()
        sims = []
        for j in range(num):
            sims.append(lst[j])
        return sims
    def output(self):
        sen = self.get_input()
        vec = self.sen2vec(self.model, sen)
        siml = self.sim(vec, self.num)
        sentences = open('diseases_out_description.txt', 'r', encoding='utf-8').readlines()
        siml_2 = []
        for i in range(len(siml)):
            siml_2.append([siml[i][1], float(100/(100+siml[i][0])), sentences[siml[i][1]].strip('\n')])
        return siml_2


# 用于疾病归类 目前使用《伤寒杂病论》目录
class NaiveBayesPredict:
    """使用训练好的模型进行预测"""
    def __init__(self, test_data_file, model_data_file):
        self.test_data = test_data_file
        self.model_data_file = open(model_data_file,'r',encoding='utf-8')
        # 每个类别的先验概率
        self.class_probabilities = {}
        # 拉普拉斯平滑，防止概率为0的情况出现
        self.laplace_smooth = 0.1
        # 模型训练结果集
        self.class_word_prob_matrix = {}
        # 当某个单词在某类别下不存在时，默认的概率（拉普拉斯平滑后）
        self.class_default_prob = {}
        # 所有单词
        self.unique_words = {}
        # 实际的新闻分类
        # self.real_classes = []
        # 预测的新闻分类
        self.predict_classes = []
    def __del__(self):
        self.model_data_file.close()
    def loadModel(self):
        # 从模型文件的第一行读取类别的先验概率
        class_probs = self.model_data_file.readline().split('#')
        for cls in class_probs:
            arr = cls.split()
            if len(arr) == 3:
                self.class_probabilities[arr[0]] = float(arr[1])
                self.class_default_prob[arr[0]] = float(arr[2])
        # 从模型文件读取单词在每个类别下的概率
        line = self.model_data_file.readline().strip()
        while len(line) > 0:
            arr = line.split()
            assert(len(arr) % 2 == 1)
            assert(arr[0] in self.class_probabilities)
            self.class_word_prob_matrix[arr[0]] = {}
            i = 1
            while i < len(arr):
                word_id = int(arr[i])
                probability = float(arr[i+1])
                if word_id not in self.unique_words:
                    self.unique_words[word_id] = 1
                self.class_word_prob_matrix[arr[0]][word_id] = probability
                i += 2
            line = self.model_data_file.readline().strip()
        # print('%d classes loaded! %d words!' %(len(self.class_probabilities), len(self.unique_words)))
    def prepare(self):
        stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
        stopwords = re.split(r'\n', stopwords)
        sentence = open(self.test_data,'r',encoding='utf-8').read()
        assert sentence != '\n'
        seg_list = list(jieba.cut(sentence))
        # 去除停用词
        seg_list_2 = []
        for w in seg_list:
            if w not in stopwords:
                seg_list_2.append(w)
        return seg_list_2
    def calculate(self):
        # 读取测试数据集
        trans = {}
        for group in open('trans.model','r',encoding='utf-8').readlines():
            lst = group.split()
            assert len(lst) == 2
            trans[lst[0]] = int(lst[1])
        line = self.prepare()
        class_score = {}
        for key in self.class_probabilities.keys():
            class_score[key] = math.log(self.class_probabilities[key])
        for word_name in line:
            if word_name not in trans:
                continue
            word_id = trans[word_name]
            if word_id not in self.unique_words:
                continue
            for class_id in self.class_probabilities.keys():
                if word_id not in self.class_word_prob_matrix[class_id]:
                    class_score[class_id] += math.log(self.class_default_prob[class_id])
                else:
                    class_score[class_id] += math.log(self.class_word_prob_matrix[class_id][word_id])
        # 对于当前新闻，所属的概率最高的分类
        max_class_score = max(class_score.values())
        for key in class_score.keys():
            if class_score[key] == max_class_score:
                return key
        return None

    def predict(self):
        self.loadModel()
        return self.calculate()


# 用于在给定的类中开方
class Medicine:
    def __init__(self, input_file, model, ids, num):
        self.inp = input_file
        self.model = model
        self.ids = ids
        self.num = num
    # 从文件中读取数据并且返回按字存储的list
    def get_input(self):
        data = []
        stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
        stopwords = re.split(r'\n', stopwords)
        for i in open(self.inp, 'r', encoding='utf-8').read():
            if i != '\n' and i not in stopwords:
                data.append(i)
        return data
    def sen2vec(self, model, sentence):
        model.random.seed(0)
        vec = model.infer_vector(sentence)
        return vec
    def sim(self, vec, num, id):
        tot_lst = np.load('anagraph.npy')
        lst = []
        for arr in tot_lst:
            if arr[-1] == id:
                lst.append(list(arr)[0:-1])
        inp_arr = np.array(vec)
        lst_arr = np.array(lst)
        lst_sim = []
        for i in range(len(lst_arr)):
            dis = np.linalg.norm(lst_arr[i] - inp_arr)
            lst_sim.append((dis, i))
        lst_sim.sort()
        sims = []
        for j in range(num):
            if j < len(lst_sim):
                sims.append(lst_sim[j])
        return sims
    def ana2print(self, ids, index):
        ana = open('yaofang.txt', 'r', encoding='utf-8').readlines()
        srch, count, cla = 0, 0, ''
        for i in range(len(ana)):
            if re.search('\d+', ana[i]):
                srch += 1
                count = 0
                cla = re.sub('\d+', '', ana[i])
            elif ids == srch:
                if i > 0 and ana[i] != '\n' and ana[i - 1] == '\n':
                    count += 1
                if index == count:
                    return ('分类： ' + cla + ana[i] + ana[i + 1] + ana[i + 2] + ana[i + 3]).strip('\n')
            elif ids < srch:
                return '\n'
        return '\n'
    def kaifang(self):
        sen = self.get_input()
        vec = self.sen2vec(self.model, sen)
        siml = self.sim(vec, self.num, self.ids)
        print(siml)
        siml_2 = []
        for sim in siml:
            siml_2.append(str(100/(100+sim[0]))+'\n'+self.ana2print(self.ids, sim[1]+1))
        return siml_2


if __name__ == '__main__':
    out_1 = open('output_1.txt', 'w', encoding='utf-8')
    out_2 = open('output_2.txt', 'w', encoding='utf-8')
    # 读取训练好的doc2vec模型
    mod = Doc2Vec.load( 'model_4.0.1.md')

    # 读取input文件，寻找最接近的方剂
    # Disease(输入文件, doc2vec模型, 输出的个数)
    # 返回list[编号 相似度 描述]
    zd = Disease('input.txt', mod, 5)
    for line in zd.output():
        print(line)
        out_1.write('Score: ' + str(line[1]) + '\n')
        out_1.write(line[2]+'\n')
        out_1.write('\n')

    # 对输入进行归类 NaiveBayesPredict(输入文件, 预先训练好的朴素贝叶斯模型即概率矩阵)
    # 返回其分类的id（目前在2-23之间）
    nbp = NaiveBayesPredict('input.txt', 'result.model')
    classify = int(nbp.predict())
    print(classify)

    # 根据分类结果在该类中进行药方的寻找
    # Medicine(输入文件, doc2vec模型, 分类id, 输出的个数)
    # 返回一个字符串list，包含相似度+开方结果
    kf = Medicine('input.txt', mod, classify, 5)
    lst = kf.kaifang()
    for line in lst:
        print(line)
        print('\n')
        out_2.write('Score: ' + line + '\n')
        out_2.write('\n')

    out_1.close()
    out_2.close()