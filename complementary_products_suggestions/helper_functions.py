from sklearn.model_selection import GroupShuffleSplit
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def train_test_split(database, test_size=0.2, single=False):
    X = database[['id_main','id_addon', 'title_main', 'title_addon']]
    if single:
        X = database[['id_main','id_addon', 'title_main', 'title_addon', 'combined']]
    y = database['label']

    tr, ts = next(GroupShuffleSplit(n_splits=1, test_size=test_size).split(X, groups = X['id_addon']))

    X_train = X.iloc[tr]
    X_test = X.iloc[ts]

    y_train = y.iloc[tr]
    y_test = y.iloc[ts]

    return X_train, X_test, y_train, y_test

def tokenize_train_test_set(X_train, X_test, max_length, single=False):
    seq = X_train['title_main'].tolist() + X_train['title_addon'].tolist()

    t = Tokenizer(lower=True, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    t.fit_on_texts(seq)

    if not single:
        train_set_main = t.texts_to_sequences(X_train["title_main"].tolist())
        train_set_main = pad_sequences(train_set_main, maxlen=max_length, padding='post')

        train_set_addon = t.texts_to_sequences(X_train["title_addon"].tolist())
        train_set_addon = pad_sequences(train_set_addon, maxlen=max_length, padding='post')

        test_set_main = t.texts_to_sequences(X_test["title_main"].tolist())
        test_set_main = pad_sequences(test_set_main, maxlen=max_length, padding='post')

        test_set_addon = t.texts_to_sequences(X_test["title_addon"].tolist())
        test_set_addon = pad_sequences(test_set_addon, maxlen=max_length, padding='post')

        return t, train_set_main, train_set_addon, test_set_main, test_set_addon

    else:
        train_set_combined = t.texts_to_sequences(X_train["combined"].tolist())
        train_set_combined = pad_sequences(train_set_combined, maxlen=60, padding='post')

        test_set_combined = t.texts_to_sequences(X_test["combined"].tolist())
        test_set_combined = pad_sequences(test_set_combined, maxlen=60, padding='post')

        return t, train_set_combined, test_set_combined