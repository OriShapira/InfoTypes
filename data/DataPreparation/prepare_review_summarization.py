import os
import json
from collections import Counter
from tqdm import tqdm
import re
from nltk import sent_tokenize
import random
import numpy as np

raw_data_folder_path = 'path/to/raw_min_10_max_100_revs/min_10_max_100_revs_filt_complete/test'
# available at https://github.com/abrazinskas/SelSum/tree/master/data
output_path = 'amasum/all_sentences.json'
output_path_sample = 'amasum/all_sentences_sample.json'

CATEGORY_CONVERSION_OPTIONS = [('book', 'Books'), ('electro', 'Electronics'), ('cloth', 'Apparel'), ('shoe', 'Apparel'),
                               ('jewl', 'Fashion'), ('bath', 'Home'), ('fashion', 'Apparel'), ('sport', 'Sports'),
                               ('home', 'Home'), ('kitchen', 'Home'), ('patio', 'Garden'), ('lawn', 'Garden'),
                               ('garden', 'Garden'), ('gadget', 'Electronics'), ('scientific', 'Scientific'),
                               ('women', 'Apparel'), ('men', 'Apparel'), ('bed', 'Home'), ('office', 'Office'),
                               ('audio', 'Audio and Video'), ('video', 'Audio and Video'), ('phone', 'Electronics'),
                               ('autom', 'Automotive'), ('car', 'Automotive'), ('vehic', 'Automotive'),
                               ('medic', 'Scientific'), ('laptop', 'Electronics'), ('food', 'Groceries'),
                               ('cook', 'Groceries'), ('vacuum', 'Vacuums'), ('keyboard', 'Electronics'),
                               ('telev', 'TV'), ('music', 'Music'), ('bluetooth', 'Electronics'),
                               ('print', 'Electronics'), ('game', 'Toys and Games'), ('toy', 'Toys and Games'),
                               ('doll', 'Toys and Games'), ('art', 'Arts and Crafts'), ('craft', 'Arts and Crafts'),
                               ('software', 'Software'), ('pet', 'Pet Supplies'), ('appliances', 'Electronics'),
                               ('computers', 'Electronics'), ('device', 'Electronics'), ('pantry', 'Groceries'),
                               ('baby', 'Baby'), ('beauty', 'Personal Care'), ('care', 'Personal Care'),
                               ('hunt', 'Sports'), ('fish', 'Sports')]

CATEGORIES_TO_USE = ['Books', 'Electronics', 'Apparel', 'Toys and Games', 'Pet Supplies']

REGEX_SPLIT_ON_PUNCTUATION = re.compile('([a-z]{2}[.!?;])([a-z]{2})', re.I)
REGEX_LINK = re.compile(r'\[\[[^\]]+\]\]')
HTML_BREAK_TAGS = ['<br/>', '<br />']


def split_review_to_sentences(review_text):
    for break_tag in HTML_BREAK_TAGS:
        review_text = review_text.replace(break_tag, ' ')  # replace line break tags with spaces
    review_text = re.sub(REGEX_LINK, '', review_text)  # remove links from the text
    review_text = re.sub(REGEX_SPLIT_ON_PUNCTUATION, r'\g<1> \g<2>', review_text)  # when there's no space after a punct
    sentences = [normalize(s) for s in sent_tokenize(review_text) if len(s) > 0]
    return sentences


def normalize(sentence):
    # “ => ", ’ => '
    res = re.sub('’', "'", sentence)
    res = re.sub('“', '"', res)
    res = re.sub('”', '"', res)
    res = re.sub('—', '-', res)
    res = ''.join(i for i in res if ord(i) < 128)
    return res


def get_category(product_categories_list):
    full_cat_str = ''
    for cat in product_categories_list:
        full_cat_str += cat.lower()

    for s, c in CATEGORY_CONVERSION_OPTIONS:
        if s in full_cat_str:
            return c

    return None


data = []
categories_counter = Counter()
asin_to_categories = {}  # asin -> [list of possible categories]
asin_to_rev_cnt = {}  # asin -> num_reviews
for filename in tqdm(os.listdir(raw_data_folder_path)):
    filepath = os.path.join(raw_data_folder_path, filename)
    asin = filename[:-5]
    with open(filepath, 'r', encoding='utf8') as fIn:
        asin_obj = json.load(fIn)
        meta_info = asin_obj['product_meta']
        if 'categories' not in meta_info:
            continue
        product_title = meta_info['title']
        # for now keep track of the possible categories for this product:
        product_categories = meta_info['categories']
        categories_counter.update(product_categories)
        asin_to_categories[asin] = product_categories
        asin_to_rev_cnt[asin] = len(asin_obj['customer_reviews'])
        # get the review sentences:
        for rev_idx, rev_obj in enumerate(asin_obj['customer_reviews']):
            rev_text = rev_obj['text']
            rev_helpful_votes = rev_obj['helpful_votes']
            rev_rating = rev_obj['rating']
            rev_id = f'review_{rev_idx}'
            rev_sentences = split_review_to_sentences(rev_text)
            for sentence_idx, sentence in enumerate(rev_sentences):
                sent_info = {
                    'asin': asin,
                    'product_title': product_title,
                    'category': '',  # will fill later
                    'helpful_votes': rev_helpful_votes,
                    'review_overall_rating': rev_rating,
                    'review_id': rev_id,
                    'sentence_idx': sentence_idx,
                    'sentence': sentence
                }
                data.append(sent_info)

        # now get the sentences in the summary (split to verdict, pros and cons):
        summary_obj = asin_obj['website_summaries'][0]
        verdict_sentences = split_review_to_sentences(summary_obj['verdict'])
        for sentence_idx, sentence in enumerate(verdict_sentences):
            sent_info = {
                'asin': asin,
                'category': '',  # will fill later
                'review_id': f'summary_verdict',
                'sentence_idx': sentence_idx,
                'sentence': sentence
            }
            data.append(sent_info)
        for summ_section in ['pros', 'cons']:
            for sentence_idx, sentence in enumerate(summary_obj[summ_section]):
                sent_info = {
                    'asin': asin,
                    'category': '',  # will fill later
                    'review_id': f'summary_{summ_section}',
                    'sentence_idx': sentence_idx,
                    'sentence': sentence
                }
                data.append(sent_info)

# for each asin, find the most frequent category label that it has out of its three:
asin_to_category = {}
asins_without_cat = {}
category_to_asins = {}
for asin in asin_to_categories:
    max_cat = get_category(asin_to_categories[asin])
    if max_cat == None:
        asins_without_cat[asin] = asin_to_categories[asin]
    asin_to_category[asin] = max_cat
    if max_cat not in category_to_asins:
        category_to_asins[max_cat] = []
    category_to_asins[max_cat].append(asin)

print('Categories used: ')
print(Counter(asin_to_category.values()))
print('---')
print('ASINs without a category found: ')
print(asins_without_cat)
print('---')
print(f'Num ASINs total: {len(asin_to_category)} (some were not assigned a category in the raw source data)')
print(f'Num sentences total: {len(data)}')
print(f'Num sentences per ASIN avg: {len(data) / len(asin_to_category)}')
print(f'Num reviews per ASIN (avg): {np.mean(list(asin_to_rev_cnt.values()))}')

# set the category for each of the sentences:
for datum in data:
    datum['category'] = asin_to_category[datum['asin']]

# save the data to a file:
print('Saving full data')
with open(output_path, 'w') as fOut:
    json.dump(data, fOut, indent=4)

# keep only a sample of the ASINs from 5 chosen categories in CATEGORIES_TO_USE:
print('Saving a sample of the data')
random.seed(100)
data_sample = []
asins_selected = set()
for cat in CATEGORIES_TO_USE:
    asin_sample = set(random.sample(category_to_asins[cat], 20))
    asins_selected.update(asin_sample)
    for datum in data:
        if datum['asin'] in asin_sample:
            data_sample.append(datum)

print('Categories used in sample: ')
print(Counter([cat for asin, cat in asin_to_category.items() if asin in asins_selected]))
print(f'Num ASINs total in sample: {len(asins_selected)}')
print(f'Num sentences total in sample: {len(data_sample)}')
print(f'Num sentences per ASIN avg in sample: {len(data_sample) / len(asins_selected)}')
print(f'Num reviews per ASIN (avg): {np.mean([c for a, c in asin_to_rev_cnt.items() if a in asins_selected])}')

with open(output_path_sample, 'w') as fOut:
    json.dump(data_sample, fOut, indent=4)

'''
100%|█████████████████████████████████████████████████████████████████████████████| 3166/3166 [00:28<00:00, 109.57it/s]
Categories used:
Counter({'Home': 722, 'Apparel': 612, 'Electronics': 597, 'Sports': 395, 'Automotive': 244, 'Garden': 147, 'Toys and Games': 96, 'Pet Supplies': 72, 'Groceries': 66, 'Baby': 46, 'Audio and Video': 40, 'Books': 39, 'Scientific': 29, 'Office': 26, 'Arts and Crafts': 10, 'Software': 9, 'Music': 1})
---
ASINs without a category found:
{}
---
Num ASINs total: 3151 (some were not assigned a category in the raw source data)
Num sentences total: 1025521
Num sentences per ASIN avg: 325.4589019358934
Num reviews per ASIN (avg): 75.83211678832117
Saving full data
Saving a sample of the data
Categories used in sample:
Counter({'Books': 20, 'Electronics': 20, 'Pet Supplies': 20, 'Apparel': 20, 'Toys and Games': 20})
Num ASINs total in sample: 100
Num sentences total in sample: 32285
Num sentences per ASIN avg in sample: 322.85
Num reviews per ASIN (avg): 77.29

Used only the first summary of each ASIN (rarely there is more than 1)

Number of review sentences total: 31574 -> 315.74 review sentences per ASIN -> 4.085 sentences per review
Number of summary sentences total: 711 -> 7.11 sentences per summary

'''