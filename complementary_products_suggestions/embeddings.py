import gensim
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

def word2vec_model(sentences):
    w2v_model = gensim.models.Word2Vec(size=300,
                                       alpha=0.03,
                                       sample=6e-5,
                                       min_alpha=0.0007,
                                       min_count = 0,
                                       workers=4)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    return w2v_model

def word2vec(content, X_train):
    sentence_list = []
    for _, row in content.iterrows():
        sentence_list.append([ word.lower() for word in row.title.split()])

    tokenizer = Tokenizer(lower=True, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    seq = X_train['title_main'].tolist() + X_train['title_addon'].tolist()
    tokenizer.fit_on_texts(seq)

    training_vocab = tokenizer.word_index
    embedding_weights = np.zeros((len(training_vocab)+1, 300))
    w2v_model = word2vec_model(sentence_list)
    for word,index in training_vocab.items():
        if word in w2v_model.wv.vocab:
            embedding_weights[index,:] = w2v_model[word]

    return embedding_weights