import json
from tqdm import tqdm
import re
from nltk import sent_tokenize
import random
from collections import Counter

raw_data_file_path = 'path/to/helpful_sentences/reviews_source/asins_for_hs_dataset.jsonl'
# The above source file is not publicly available, and was acquired with the assistance of the authors
# of the helpful sentence paper.
# An alternative can be to get similar data from the publicly available Amazon Review Dataset.
categories_data_file_path = 'helpful_sentences_asin_categories.jsonl'
output_path = 'Sentiment/all_sentences.json'
output_path_sample = 'Sentiment/all_sentences_sample.json'

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


asin_to_info = {}
with open(categories_data_file_path, 'r') as fIn:
    for line_idx, line in enumerate(tqdm(fIn)):
        asin_data = json.loads(line.strip())
        asin = asin_data['asin']
        item_name = asin_data['item_name']
        category = asin_data['gl_product_group_desc']
        if category.startswith('gl_'):
            category = category[3:]
        asin_to_info[asin] = {'category': category, 'item_name': item_name}

data = []
review_ids = set()
with open(raw_data_file_path, 'r', encoding='utf8') as fIn:
    for line_idx, line in enumerate(tqdm(fIn)):
        line = line.strip()
        asin_data = json.loads(line)
        for review_data in asin_data['reviews']:
            if 'overall_rating' not in review_data:
                continue
            asin = review_data['asin']
            review_text = review_data['text']
            overall_rating = review_data['overall_rating']
            review_id = review_data['review_id']
            helpful_count = review_data['helpful_count'] if 'helpful_count' in review_data else None
            nothelpful_count = review_data['nothelpful_count'] if 'nothelpful_count' in review_data else None
            sentences = split_review_to_sentences(review_text)
            for sentence_idx, sentence in enumerate(sentences):
                sent_info = {
                    'asin': asin,
                    'category': asin_to_info[asin]['category'],
                    'review_helpful_count': helpful_count,
                    'review_nothelpful_count': nothelpful_count,
                    'review_overall_rating': overall_rating,
                    'review_id': review_id,
                    'sentence_idx': sentence_idx,
                    'sentence': sentence
                }
                data.append(sent_info)
            if len(sentences) > 0:
                review_ids.add(review_id)


# get a sample of reviews for the different levels of helpfulness votes:
random.seed(100)
review_ids_sample = random.sample(list(review_ids), 5000)
print(f'Num revs total: {len(review_ids)} vs sample: {len(review_ids_sample)}\n')
data_sample = []
for datum in data:
    if datum['review_id'] in review_ids_sample:
        data_sample.append(datum)

with open(output_path_sample, 'w') as fOut:
    json.dump(data_sample, fOut, indent=4)


def count_classes(review_ids_to_assess):
    revid_to_score = {}
    revid_to_sentcount = Counter()
    for d in data:
        rev_id = d['review_id']
        if rev_id in review_ids_to_assess:
            if rev_id not in revid_to_score:
                revid_to_score[rev_id] = d['review_overall_rating']
            revid_to_sentcount[rev_id] += 1

    counter = Counter(revid_to_score.values())
    print('Score distribution:')
    print(counter)

    num_sents = sum(revid_to_sentcount.values())
    num_reviews = len(revid_to_sentcount)
    sents_per_rev = num_sents / num_reviews
    print('Counts:')
    print(f'Num Sentences Total: {num_sents}')
    print(f'Num Reviews Total: {num_reviews}')
    print(f'Sents per Review overall: {sents_per_rev}')


print('Full stats:')
count_classes(review_ids)
print()
print('Sample Stats:')
count_classes(review_ids_sample)


'''
123it [00:00, 120228.24it/s]
123it [00:07, 17.51it/s]
Num revs total: 58205 vs sample: 5000

Full stats:
Score distribution:
Counter({5: 39215, 4: 7708, 1: 5062, 3: 3805, 2: 2415})
Counts:
Num Sentences Total: 238251
Num Reviews Total: 58205
Sents per Review overall: 4.093308135039945

Sample Stats:
Score distribution:
Counter({5: 3337, 4: 675, 1: 448, 3: 321, 2: 219})
Counts:
Num Sentences Total: 20262
Num Reviews Total: 5000
Sents per Review overall: 4.0524
'''