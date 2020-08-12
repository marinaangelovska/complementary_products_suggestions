import random
import pandas as pd
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
import re
import string

def generate_negative_samples_two_conditions(content, matches, bought_together, n):
    counter = tqdm()

    distribution_addon = get_addon_distribution(matches)
    for item in distribution_addon:
        globalid_addon = item[0]
        positive_samples = item[1]
        negative_counter = 0
        while negative_counter < positive_samples:
            random1 = random.sample(range(0, len(content) - 1), 1)
            globalid_main = content.iloc[random1]['id'].iloc[0]

            occurences_in_matches = ((matches.globalid_main == globalid_main) & (matches.globalid_addon == globalid_addon)).any()
            occurences_in_bought_together = ((bought_together.globalid_left == globalid_main) & (bought_together.globalid_right == globalid_addon)).any()

            if not occurences_in_matches and not occurences_in_bought_together:
                matches = matches.append({"id_main": globalid_main,
                                          "id_addon": globalid_addon,
                                          'label': int(0)}, ignore_index=True)
                negative_counter += 1
                counter.update(1)


    distribution_main = get_main_distribution(matches)
    for item in distribution_main:
        globalid_main = item[0]
        positive_samples = item[1]
        negative_samples = item[2]
        while negative_samples <= positive_samples:
            random1 = random.sample(range(0, len(content) - 1), 1)
            globalid_addon = content.iloc[random1]['id'].iloc[0]

            occurences_in_matches = ((matches.globalid_main == globalid_main) & (matches.globalid_addon == globalid_addon)).any()
            occurences_in_bought_together = ((bought_together.globalid_left == globalid_main) & (bought_together.globalid_right == globalid_addon)).any()

            if not occurences_in_matches and not occurences_in_bought_together:
                matches = matches.append({"id_main": globalid_main,
                                          "id_addon": globalid_addon,
                                          'label': int(0)}, ignore_index=True)
                negative_samples += 1
                counter.update(1)

    counter.close()
    return matches


def get_addon_distribution(matches):
    unique_ids = set(matches['id_addon'])
    distribution = []
    for unique_id in unique_ids:
        positive = len(matches[matches['id_addon'] == unique_id].iloc[:, [0, 1, 2]].values)
        distribution.append((unique_id, positive))

    return distribution


def get_main_distribution(matches):
    unique_ids = set(matches['id_main'])
    distribution = []
    for unique_id in unique_ids:
        positive = 0
        negative = 0
        values = matches[matches['id_main'] == unique_id].iloc[:, [2]].values
        for x in values:
            if x == 1:
                positive += 1
            elif x == 0:
                negative += 1

        distribution.append((unique_id, positive, negative))

    return distribution

def merge_two_datasets(content, matches, columns, left, right):
    return pd.merge(matches, content[columns], left_on=left, right_on=right, suffixes=('_main', '_addon'))

def concatanate_columns(df, columns):
   return df[columns].apply(lambda x: str(x[0])+" "+str(x[1]), axis=1)


def clean_text(text):
    stop_words = stopwords.words('dutch')
    meaningless_words = ['cm', 'm', 'ml', 'l', 'dcl']
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    text_no_digits = ' '.join(s for s in text_nopunct.split() if not any(c.isdigit() for c in s))
    text_no_one_letter_words = ' '.join( [w.lower() for w in text_no_digits.split() if len(w)>1] )
    text_no_stop_words = ' '.join( [w for w in text_no_one_letter_words.split() if not w.lower() in stop_words] )
    text_no_meaningless_words = ' '.join( [w for w in text_no_stop_words.split() if not w.lower() in meaningless_words] )
    return text_no_meaningless_words