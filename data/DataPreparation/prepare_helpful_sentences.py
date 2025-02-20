import os
import json
from tqdm import tqdm

# raw_data_file_path = 'path/to/helpful_sentences/test.json'
raw_data_file_path = 'path/to/helpful_sentences/train.json'
# available at https://registry.opendata.aws/helpful-sentences-from-reviews/
categories_data_file_path = 'helpful_sentences_asin_categories.jsonl'
# output_path = 'HelpfulSentences/all_sentences.json'
output_path = 'HelpfulSentences/all_sentences_train.json'

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
with open(raw_data_file_path, 'r', encoding='utf8') as fIn:
    for line_idx, line in enumerate(tqdm(fIn)):
        asin_data = json.loads(line.strip())
        asin = asin_data['asin']
        sent_info = {
            'asin': asin_data['asin'],
            'product_title': asin_data['product_title'],
            'helpful_score': asin_data['helpful'],
            'sentence_idx': f'{line_idx}',
            'category': asin_to_info[asin]['category'],
            'sentence': asin_data['sentence']
        }
        data.append(sent_info)

with open(output_path, 'w') as fOut:
    json.dump(data, fOut, indent=4)