import json
from tqdm import tqdm
import re
from nltk import sent_tokenize

raw_data_file_path = 'path/to/helpful_sentences/reviews_source/asins_for_hs_dataset.jsonl'
# The above source file is not publicly available, and was acquired with the assistance of the authors
# of the helpful sentence paper.
# An alternative can be to get similar data from the publicly available Amazon Review Dataset.
categories_data_file_path = 'helpful_sentences_asin_categories.jsonl'
output_path = 'HelpfulReviews/all_sentences.json'
helpfulness_output_path = 'HelpfulReviews/all_sentences_sample.json'

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
review_ids_no_helpful = []
review_ids_very_helpful = []
review_ids_not_helpful = []
with open(raw_data_file_path, 'r', encoding='utf8') as fIn:
    for line_idx, line in enumerate(tqdm(fIn)):
        line = line.strip()
        asin_data = json.loads(line)
        for review_data in asin_data['reviews']:
            if 'helpful_count' not in review_data:
                continue
            asin = review_data['asin']
            review_text = review_data['text']
            overall_rating = review_data['overall_rating']
            review_id = review_data['review_id']
            helpful_count = review_data['helpful_count']
            nothelpful_count = review_data['nothelpful_count']
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
            # keep review_ids by helpfulness votes:
            if helpful_count >= 9 and nothelpful_count <= 0:
                review_ids_very_helpful.append(review_id)
            elif helpful_count == 0 and nothelpful_count <= 1:
                review_ids_no_helpful.append(review_id)
            elif nothelpful_count >= 3 and helpful_count == 0:
                review_ids_not_helpful.append(review_id)

review_ids_no_helpful_sample = set()
review_ids_very_helpful_sample = set(review_ids_very_helpful)
review_ids_not_helpful_sample = set(review_ids_not_helpful)
data_no_helpful = []
data_very_helpful = []
data_not_helpful = []
for datum in data:
    if datum['review_id'] in review_ids_no_helpful_sample:
        data_no_helpful.append(datum)
    elif datum['review_id'] in review_ids_very_helpful_sample:
        data_very_helpful.append(datum)
    elif datum['review_id'] in review_ids_not_helpful_sample:
        data_not_helpful.append(datum)

with open(helpfulness_output_path, 'w') as fOut:
    json.dump(data_very_helpful + data_no_helpful + data_not_helpful, fOut, indent=4)

print(f'helpful reviews: {len(review_ids_very_helpful)} -> {len(review_ids_very_helpful_sample)} ({len(data_very_helpful)} sentences)')
print(f'nohelpful reviews: {len(review_ids_no_helpful)} -> {len(review_ids_no_helpful_sample)} ({len(data_no_helpful)} sentences)')
print(f'unhelpful reviews: {len(review_ids_not_helpful)} -> {len(review_ids_not_helpful_sample)} ({len(data_not_helpful)} sentences)')


'''
123it [00:00, 119338.28it/s]
123it [00:04, 29.25it/s]
helpful reviews: 458 -> 458 (4969 sentences)
nohelpful reviews: 2987 -> 0 (0 sentences)
unhelpful reviews: 486 -> 486 (1898 sentences)

helpful_count >= 9 and nothelpful_count <= 0 - all used
helpful_count == 0 and nothelpful_count <= 1 - not used
nothelpful_count >= 3 and helpful_count == 0 - all used

thresholds used so that there are <= 500 of each of the two classes
'''