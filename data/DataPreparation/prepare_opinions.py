import json
from tqdm import tqdm
from nltk import sent_tokenize
import numpy as np
from collections import Counter

raw_data_train_path = 'path/to/AM2/training.jsonlist'
raw_data_test_path = 'path/to/AM2/test.jsonlist'
# available at https://facultystaff.richmond.edu/~jpark/data/am2_emnlp2022.zip
output_path = 'AM2/all_sentences.json'

data = []
asin_to_review_ids = {}  # asin -> set{review_ids}
sents_per_subset = Counter()
for filepath, split in [(raw_data_train_path, 'train'), (raw_data_test_path, 'test')]:
    with open(filepath, 'r') as fIn:
        for line in tqdm(fIn):
            review_obj = json.loads(line.strip())
            review_id = review_obj['reviewID']
            asin = review_obj['asin']
            review_rating = review_obj['overall']
            review_helpful_votes = int(review_obj['vote']) if 'vote' in review_obj else 0
            if asin not in asin_to_review_ids:
                asin_to_review_ids[asin] = set()
            asin_to_review_ids[asin].add(review_id)

            # get the propositions of this review:
            prop_id_to_text = {}  # prop_id -> prop_text
            review_texts = []  # [<all_proposition_texts>]
            prop_texts_subj = []  # [(prop_id, prop_text)] - only for the subjective texts and keeping the order
            prop_id_to_reasons = {}  # prop_id -> [prop_ids] - only for the subjective texts
            for prop in review_obj['propositions']:
                prop_id = prop['id']
                prop_type = prop['type']
                prop_text = prop['text']
                prop_id_to_text[prop_id] = prop_text
                review_texts.append(prop_text)

                if prop_type in ['value',
                                 'policy']:  # only the subjective types ('fact' and 'testimony' and 'reference' are objective types)
                    # get the proposition IDs for the reason (list of ints or range representations):
                    prop_reasons = prop['reasons']
                    prop_reasons_ids = []
                    if prop_reasons is not None:
                        for reason in prop_reasons:
                            if '_' in reason:  # in case of inclusive range (e.g. "4_6")
                                ids = list(map(int, reason.split('_')))
                                prop_reasons_ids.extend(list(range(ids[0], ids[1] + 1)))
                            else:
                                prop_reasons_ids.append(int(reason))

                    prop_id_to_reasons[prop_id] = prop_reasons_ids
                    prop_texts_subj.append((prop_id, prop_text))

            # convert the information to the sentence level:
            review_full_text = ' '.join(review_texts)
            review_sentences = sent_tokenize(review_full_text)
            used_props = set()
            for sent_idx, sent in enumerate(review_sentences):
                sent_is_opinion = False
                sent_has_reason_for_opinion = False
                for prop_id, prop_text in prop_texts_subj:
                    # do not look for a proposition that was already found before:
                    if prop_id in used_props:
                        continue
                    # check if the propoisition is in the current sentence:
                    if prop_text in sent:
                        sent_is_opinion = True
                        used_props.add(prop_id)
                        # check if a reason is also included in the current sentence:
                        for reason_prop_id in prop_id_to_reasons[prop_id]:
                            reason_text = prop_id_to_text[reason_prop_id]
                            if reason_text in sent:
                                sent_has_reason_for_opinion = True

                sent_info = {
                    'asin': asin,
                    'review_id': review_id,
                    'sentence_idx': sent_idx,
                    'category': 'headphones',
                    'review_helpful_votes': review_helpful_votes,
                    'review_overall_rating': review_rating,
                    'is_opinion': int(sent_is_opinion),
                    'is_opinion_with_reason': int(sent_has_reason_for_opinion),
                    'split': split,
                    'sentence': sent
                }
                data.append(sent_info)
                sents_per_subset[split] += 1

print(f'Num ASINs total: {len(asin_to_review_ids)}')
print(f'Num sentences total: {len(data)}')
print(f'Num sentences per ASIN avg: {len(data) / len(asin_to_review_ids)}')
print(f'Num reviews per ASIN (avg): {np.mean(list(map(len, asin_to_review_ids.values())))}')
for subset in sents_per_subset:
    print(f'Num sentences in subset {subset}: {sents_per_subset[subset]}')

# save the data to a file:
print('Saving full data')
with open(output_path, 'w') as fOut:
    json.dump(data, fOut, indent=4)


'''
Num ASINs total: 693
Num sentences total: 4491
Num sentences per ASIN avg: 6.48051948051948
Num reviews per ASIN (avg): 1.266955266955267
Num sentences in subset train: 3105
Num sentences in subset test: 1386
Saving full data

sentence type count:
is_opinion total 3132 (2133 train, 999 test)
of which is_opinion_with_reason total 363 (249 train, 114 test)
not opinions total 1359 (972 train, 387 test)


Note that the sentences in the AM2 dataset are all argumentative, and therefore the data here does not have any 
non-argumentative parts from reviews.
'''