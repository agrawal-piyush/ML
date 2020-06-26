from flask import Flask,request,render_template
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle





main = Flask(__name__)

@main.route('/')
def home():
    return render_template('home.html')
@main.route('/form')
def form():
    return render_template('form.html')

@main.route('/score',methods=['POST'])
def score():

    def get_count_vectors(essays):

        vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')

        count_vectors = vectorizer.fit_transform(essays)

        feature_names = vectorizer.get_feature_names()

        return feature_names, count_vectors

    def sentence_to_wordlist(raw_sentence):

        clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
        tokens = nltk.word_tokenize(clean_sentence)

        return tokens

    def tokenize(essay):
        stripped_essay = essay.strip()

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(stripped_essay)

        tokenized_sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

        return tokenized_sentences

    def avg_word_len(essay):

        clean_essay = re.sub(r'\W', ' ', essay)
        words = nltk.word_tokenize(clean_essay)

        return sum(len(word) for word in words) / len(words)

    def word_count(essay):

        clean_essay = re.sub(r'\W', ' ', essay)
        words = nltk.word_tokenize(clean_essay)

        return len(words)

    def char_count(essay):

        clean_essay = re.sub(r'\s', '', str(essay).lower())

        return len(clean_essay)

    def sent_count(essay):

        sentences = nltk.sent_tokenize(essay)

        return len(sentences)
    def count_lemmas(essay):

        tokenized_sentences = tokenize(essay)

        lemmas = []
        wordnet_lemmatizer = WordNetLemmatizer()

        for sentence in tokenized_sentences:
            tagged_tokens = nltk.pos_tag(sentence)

            for token_tuple in tagged_tokens:

                pos_tag = token_tuple[1]

                if pos_tag.startswith('N'):
                    pos = wordnet.NOUN
                    lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                elif pos_tag.startswith('J'):
                    pos = wordnet.ADJ
                    lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                elif pos_tag.startswith('V'):
                    pos = wordnet.VERB
                    lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                elif pos_tag.startswith('R'):
                    pos = wordnet.ADV
                    lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
                else:
                    pos = wordnet.NOUN
                    lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

        lemma_count = len(set(lemmas))

        return lemma_count
    def count_spell_error(essay):

        clean_essay = re.sub(r'\W', ' ', str(essay).lower())
        clean_essay = re.sub(r'[0-9]', '', clean_essay)
        data1 = open('big.txt').read()
        words_ = re.findall('[a-z]+', data1.lower())
        word_dict = collections.defaultdict(lambda: 0)
        for word in words_:
            word_dict[word] += 1
        clean_essay = re.sub(r'\W', ' ', str(essay).lower())
        clean_essay = re.sub(r'[0-9]', '', clean_essay)

        mispell_count = 0

        words = clean_essay.split()

        for word in words:
            if not word in word_dict:
                mispell_count += 1

        return mispell_count

    def count_pos(essay):

        tokenized_sentences = tokenize(essay)

        noun_count = 0
        adj_count = 0
        verb_count = 0
        adv_count = 0

        for sentence in tokenized_sentences:
            tagged_tokens = nltk.pos_tag(sentence)

            for token_tuple in tagged_tokens:
                pos_tag = token_tuple[1]

                if pos_tag.startswith('N'):
                    noun_count += 1
                elif pos_tag.startswith('J'):
                    adj_count += 1
                elif pos_tag.startswith('V'):
                    verb_count += 1
                elif pos_tag.startswith('R'):
                    adv_count += 1

        return noun_count, adj_count, verb_count, adv_count



    def extract_features(data):

        features = data.copy()

        features['char_count'] = features['essay'].apply(char_count)

        features['word_count'] = features['essay'].apply(word_count)

        features['sent_count'] = features['essay'].apply(sent_count)

        features['avg_word_len'] = features['essay'].apply(avg_word_len)

        features['lemma_count'] = features['essay'].apply(count_lemmas)

        features['spell_err_count'] = features['essay'].apply(count_spell_error)

        features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))

        return features

    line1 = request.form['t1']
    line2 = request.form['t2']
    line3 = request.form['t3']
    line4 = request.form['t4']
    line5 = request.form['t5']
    line6 = request.form['t6']
    line7 = request.form['t7']
    line8 = request.form['t8']
    line9 = request.form['t9']
    line10 = request.form['t10']

    import csv
    with open('temp.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["essay_id","essay_set","essay"])
        writer.writerow([1,1,line1])
        writer.writerow([2,1,line2])
        writer.writerow([3,1,line3])
        writer.writerow([4,1,line4])
        writer.writerow([5,1,line5])
        writer.writerow([6,1,line6])
        writer.writerow([7,1,line7])
        writer.writerow([8,1,line8])
        writer.writerow([9,1,line9])
        writer.writerow([10,1,line10])

    df = pd.read_csv("temp.csv")
    model = pickle.load(open('model1.sav','rb'))

    f_set = extract_features(df[df['essay_set'] == 1])

    final_features = f_set.iloc[:, 3:].as_matrix()
    prediction = model.predict(final_features)
    score = 0
    for i in range(10):
        score += prediction[i]
    f_score= int(score/10)
    return render_template("result.html",value = f_score)
main.run()
