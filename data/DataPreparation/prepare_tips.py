import json
import csv
from tqdm import tqdm
import random

raw_data_file_path = 'path/to/generate_tips_data.tsv'
# available at http://proj.ise.bgu.ac.il/public/gen_tips.zip
output_path = 'tips/all_sentences.json'
output_path_sample = 'tips/all_sentences_sample.json'

data = []
categories = set()
with open(raw_data_file_path, 'r', encoding='utf8') as fIn:
    rows = csv.DictReader(fIn, delimiter="\t", quotechar='"')
    for row_idx, row in enumerate(tqdm(rows)):
        sent_info = {
            'asin': row['asin'].strip(),
            'category': row['category'].strip(),
            'review_id': row['review_id'].strip(),
            'sentence_idx': row['num_sentence'].strip(),
            'sentence': row['sentence'].strip(),
            'is_tip': row['tip'].strip(),  # 1 | 0
            'tip_type': row['type'].strip(),  # <string>
            'tip_before_after': row['before/after'].strip(),  # before | after | both
            'reviewer_id': row['reviewer_id'].strip(),
            'is_standalone': row['standalone'].strip(),  # yes | no
            'extend_position': row['extend_position'].strip()  # before | after | both
        }
        data.append(sent_info)
        categories.add(row['category'].strip())

with open(output_path, 'w') as fOut:
    json.dump(data, fOut, indent=4)

# get all the tips, and a sample of the non-tips (twice as many as the tips):
data_tips = [d for d in data if d['is_tip'] == '1']
data_not_tips = [d for d in data if d['is_tip'] == '0']

random.seed(100)
data_not_tips_sample = random.sample(data_not_tips, 2 * len(data_tips))

with open(output_path_sample, 'w') as fOut:
    json.dump(data_tips + data_not_tips_sample, fOut, indent=4)

print(f'Num total: {len(data)}')
print(f'Num tips: {len(data_tips)}')
print(f'Num not tips: {len(data_not_tips)}')
print(f'Num not tips sample: {len(data_not_tips_sample)}')
print(f'Categories: {categories}')


'''
OUTPUT SHOULD BE:
Num total: 85171
Num tips: 3848
Num not tips: 81323
Num not tips sample: 7696    -> this is a sample of the non-tips (twice as many as tips)
Categories: {'Musical Instruments', 'Baby', 'Toys & Games', 'Tools & Home Improvement', 'Sports & Outdoors'}
'''