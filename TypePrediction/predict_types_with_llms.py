'''
This script runs an LLM to predict the typology types for a given json with sentences.
The input file can be created with one of the scripts in data/DataPreparation. The input format expects:
[
    {
        "asin": <str>,
        "category": <str>,
        "review_id": <str>,
        "sentence_idx": <int>,
        "sentence": <str>>
    }, ...
]
The output is a json file with prediction scores for the types in the typology.

To run:
- set INPUT_GOLD_ANNOTATION_PATH in the script
- set OUTPUT_PATH in the script
- set MODELS_TO_USE  in the script (though the default is likely what you want - flan-t5-xxl)
- run python annotate_test_with_llms.py
'''

import csv
import json
import torch
import os
import numpy as np
from tqdm import tqdm
import time

# model name and temperature to use with it
MODELS_TO_USE = {'flan_t5_xxl': 0.3} #, 'flan_t5_xl': 0.3, 'flan_t5_large': 0.3, 'flan_t5_base': 0.3, 'flan_t5_snall': 0.3, 'flan_ul2': 0.3} #'random': 0, 'j2_jumbo_instruct': 0.7} 'flan_ul2': 0.3


if 'flan_t5_xxl' in MODELS_TO_USE:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
elif 'flan_ul2' in MODELS_TO_USE:
    from transformers import T5ForConditionalGeneration, AutoTokenizer
elif 'j2_jumbo_instruct' in MODELS_TO_USE:
    import ai21
    ai21.api_key = 'XYZ'  # API key is in https://studio.ai21.com/account
    

# Create an input file with one of the scripts under data/DataPreparation
INPUT_GOLD_ANNOTATION_PATH = 'path/to/sentences_json_file'
OUTPUT_TEMP_INTERMEDIATE_SAVE_PATH = 'tmp_data.json'
OUTPUT_PATH = 'results.json'
MAX_TOKENS = 50
NUM_RESPONSES_PER_INSTANCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the sentence types with their prompt substrings
SENTENCE_TYPES = {
    'opinion': 'does the sentence express an opinion about anything',
    'opinion_with_reason': 'does the sentence express an opinion about anything and also provide reasoning for it',
    'personal_info': 'does the sentence say something about someone',
	'improvement_desire': 'does the sentence say how the product could be improved',
	'personal_usage': 'does the sentence describe how someone used the product',
    'setup': 'does the sentence describe something about the setup or installation of the product',
	'product_usage': 'does the sentence describe how the product can be used',
	'situation': 'does the sentence describe a condition under which the product is used',
	'product_description': 'does the sentence describe something objective about the product like its characteristics or its plot',
	'tip': 'does the sentence provide a tip on the product',
	'speculative': 'does the sentence speculate about something',
	'comparative': 'does the sentence compare to another product',
	'comparative_general': 'does the sentence describe something that compares the product generally to something that is not a product',
	'compatibility': 'does the sentence describe the compatibility of the product with another product',
	'price': 'does the sentence explicitly talk about the price of the product',
	'buy_decision': 'does the sentence explicitly talk about buying the product',
	'general_info': 'does the sentence describe general information that is not necesarilly in regards to the product',
    'comparative_seller': 'does the sentence compare between sellers of the product',
	'seller_experience': 'does the sentence describe something about the experience with the seller',
	'delivery_experience': 'does the sentence describe the shipment of the product',
	'imagery': 'is the sentence written in a figurative style',
	'sarcasm': 'does the sentence contain a sarcastic expression',
	'rhetorical': 'is the sentence rhetorical or used as a filler or for transition without any real value',
	'inappropriate': 'does the sentence contain content that is toxic or unnecessarily racy'
}

# the category strings (which vary across the datasets) and their prompt substring
CATEGORY_STRS = {
    'electronics': 'an electronics product',
    'camera': 'a camera product',
    'dvd': 'a dvd',
    'music': 'a music product',
    'book': 'a book',
    'toy': 'a toy',
    'bags_and_cases': 'a bag or case',
    'bluetooth': 'a bluetooth product',
    'boots': 'boots',
    'keyboards': 'a keyboard product',
    'tv': 'a television product',
    'vacuums': 'a vacuum product',
    'AMAZON_FASHION': 'a fashion product',
    'Automotive': 'an automotive product',
    'Books': 'a book',
    'CDs_and_Vinyl': 'a music disc',
    'Digital_Music': 'a digital music product',
    'Electronics': 'an electronics product',
    'Movies_and_TV': 'a movie or TV show',
    'Toys_and_Games': 'a toy or game',
    'Pet Supplies': 'a pet supplies product',
    'Apparel': 'an apparel product',
    'Toys and Games': 'a toy or game',
    'Toys & Games': 'a toy or game',
    'Tools & Home Improvement': 'a home improvement product',
    'Baby': 'a baby product',
    'Sports & Outdoors': 'a sports or outdoors product',
    'Musical Instruments': 'a musical instrument',
    'headphones': 'headphones',
    'home': 'a home product',
    'clothing': 'a fashion product'
}


class Model_base:
    def load(self):
        raise NotImplementedError()
        
    def unload(self):
        raise NotImplementedError()
        
    def infer_instance(self, sentence_str, category_str, sent_type_question_str):
        raise NotImplementedError()
        
    def infer_instance(self, sentence_str, category_str, sent_type_question_str):
        raise NotImplementedError()
        
    def infer(self, data, output_temp_intermediate_save_path):
        missed_instances = []
        # inference probabilities are returned within the given data dictionary
        for instance_idx, instance in enumerate(tqdm(data)):
            try:
                sentence_str = instance['sentence']
                category_str = CATEGORY_STRS[instance['category']]
                # the probabilities might have been computed already (from intermediate temp file):
                if 'sentence_types' in instance and self.name in instance['sentence_types']:
                    continue
                
                sentence_type_to_probability = {}
                # for each sentence type, predict the probability that it is relevant for the current sentence instance:
                for sent_type, sent_type_question_str in SENTENCE_TYPES.items():
                    probability = self.infer_instance(sentence_str, category_str, sent_type_question_str)
                    if probability > 0:
                        sentence_type_to_probability[sent_type] = probability
                # set the probabilities for this instance in the 'data' data structure:
                instance['sentence_types'][self.name] = sentence_type_to_probability
            except Exception as e:
                missed_instances.append(instance_idx)
                raise e
            
            if instance_idx % 10 == 0:
                save_data(data, output_temp_intermediate_save_path)
        
        print(f'Failed on {len(missed_instances)} instances.')
        return missed_instances


class Model_flan_t5(Model_base):
    def __init__(self, model_name, temperature=0.3):
        self.name = model_name  #'flan_t5_xxl'
        self.model_name = f'google/{model_name.replace("_", "-")}'  #flan-t5-xxl'
        self.seed = 42
        self.model = None
        self.accept_threshold = 0.5
        self.temperature = temperature
        self.prompt = 'Given that this sentence is from a product review about {category}, {sent_type_question}? Answer yes or no. The sentence is: "{sentence}"'

    def load(self):
        self.set_seed()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16, load_in_8bit=False)#.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def set_seed(self):
        # set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def unload(self):
        # remove the model from memory to allow other models needed next
        del self.model
        torch.cuda.empty_cache()
        
    def infer_instance(self, sentence_str, category_str, sent_type_question_str):
        # prepare the prompt:
        prompt_str = self.prompt.format(category=category_str, sent_type_question=sent_type_question_str, sentence=sentence_str)
        # tokenize the prompt:
        input_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids
        # generate the output(s):
        responses_tokens = self.model.generate(
            input_ids.to(DEVICE),
            do_sample=True,
            temperature=self.temperature,
            max_length=MAX_TOKENS,
            num_return_sequences=NUM_RESPONSES_PER_INSTANCE)
        # detokenize the response(s):
        responses_texts = [self.tokenizer.decode(response_tokens, skip_special_tokens=True) for response_tokens in responses_tokens]
        # get the percentage of responses that have a "yes" in them,
        # used as the probability that the model predicts that the sentence type is relevant to the sentence:
        num_yes = sum([1 if 'yes' in r.lower() else 0 for r in responses_texts])
        prob = num_yes / len(responses_texts)
        return prob  # prob if prob >= self.accept_threshold else 0.
        
        
class Model_flan_ul2(Model_flan_t5):
    def __init__(self, temperature=0.3):
        self.name = 'flan_ul2'
        self.model_name = 'google/flan-ul2'
        self.seed = 42
        self.model = None
        self.accept_threshold = 0.5
        self.temperature = temperature
        self.prompt = 'Given that this sentence is from a product review about {category}, {sent_type_question}? Answer yes or no. The sentence is: "{sentence}"'
        
    def load(self):
        self.set_seed()
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        
        
class Model_j2_jumbo_instruct(Model_base):
    def __init__(self, temperature=0.7):
        self.name = 'j2_jumbo_instruct'
        self.model_name = 'j2-jumbo-instruct'
        self.accept_threshold = 0.5
        self.temperature = temperature
        self.prompt = 'Given that this sentence is from a product review about {category}, {sent_type_question}? Answer yes or no and explain. The sentence is: "{sentence}"'
        
    def load(self):
        pass

    def unload(self):
        pass
        
    def infer_instance(self, sentence_str, category_str, sent_type_question_str):
        prompt_str = self.prompt.format(category=category_str, sent_type_question=sent_type_question_str, sentence=sentence_str)
        resp = ai21.Completion.execute(
            model=self.model_name,
            prompt=prompt_str,
            numResults=NUM_RESPONSES_PER_INSTANCE,
            maxTokens=MAX_TOKENS,
            temperature=self.temperature,
            topKReturn=0,
            topP=1,
            countPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            frequencyPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            presencePenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
            },
            stopSequences=[]
        )
        # extract the response(s):
        responses_texts = [resp["data"]["text"].strip() for resp in resp['completions']]
        # get the percentage of responses that have a "yes" in them,
        # used as the probability that the model predicts that the sentence type is relevant to the sentence:
        num_yes = sum([1 if 'yes' in r else 0 for r in responses_texts])
        prob = num_yes / len(responses_texts)
        return prob  # prob if prob >= self.accept_threshold else 0.


class Model_random(Model_base):
    def __init__(self, temperature=0.5):
        self.name = 'random'
        self.accept_threshold = 0.5
        
    def load(self):
        pass

    def unload(self):
        pass
        
    def infer_instance(self, sentence_str, category_str, sent_type_question_str):
        prob = sum(np.random.randint(2, size=NUM_RESPONSES_PER_INSTANCE)) / NUM_RESPONSES_PER_INSTANCE
        return prob  # prob if prob >= self.accept_threshold else 0.


def get_test_set_data(input_gold_annotations_path, output_temp_intermediate_save_path):
    # if there's an intermediate temp file already saved, use it:
    if os.path.exists(output_temp_intermediate_save_path):
        print(f'WARNING: Using data from temp file: {output_temp_intermediate_save_path}')
        with open(output_temp_intermediate_save_path) as fIn:
            data = json.load(fIn)
    else:
        data = []
        with open(input_gold_annotations_path, newline='') as fIn:
            reader = csv.DictReader(fIn)
            for row in reader:
                gold_labels = {l: 1. for l in row['what_types_are_relevant_for_the_bold_sentence_check_one_or_more_gold'].split()}
                asin = row['asin']
                category = row['category']
                review_id = row['review_id']
                sentence = row['sentence']
                instance_id = row['_id']
                data.append({'asin': asin, 
                             'category': category,
                             'review_id': review_id,
                             'instance_id': instance_id,
                             'sentence': sentence,
                             'sentence_types': {'gold': gold_labels}})
    return data
    

def get_annotation_set_data(input_annotations_path, output_temp_intermediate_save_path):
    # if there's an intermediate temp file already saved, use it:
    if os.path.exists(output_temp_intermediate_save_path):
        print(f'WARNING: Using data from temp file: {output_temp_intermediate_save_path}')
        with open(output_temp_intermediate_save_path) as fIn:
            data = json.load(fIn)
    else:
        data = []
        with open(input_annotations_path, 'r') as fIn:
            data = json.load(fIn)
            for datum in data:
                datum['sentence_types'] = {}
    return data


def save_data(data, output_path):
    with open(output_path, 'w') as fOut:
        json.dump(data, fOut, indent=4)


def compute_final(models, data):
    model_names = [m.name for m in models]
    results = {}
    
    if 'gold' in data[0]['sentence_types']:
        # for each instance, compute the F1 of the labels between the model prediction and the gold labels
        # of the instance, this is the score of the instance for the model:
        for instance in data:
            if 'f1' not in instance:
                instance['f1'] = {}
                instance['recall'] = {}
                instance['precision'] = {}
            gold_labels = instance['sentence_types']['gold']
            for model in models:
                if model.name not in instance['sentence_types']:
                    continue  # the model may have failed to get results for this instance, so skip it
                model_label_probs = {l: p for l, p in instance['sentence_types'][model.name].items() if p >= model.accept_threshold}
                tp = [l for l in gold_labels if l in model_label_probs]
                fp = [l for l in model_label_probs if l not in gold_labels]
                recall = len(tp) / len(gold_labels)
                precision = len(tp) / len(model_label_probs) if len(model_label_probs) > 0 else 0.
                f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.
                instance['f1'][model.name] = f1
                instance['recall'][model.name] = recall
                instance['precision'][model.name] = precision
                
        avg_f1 = {}
        avg_recall = {}
        avg_precision = {}
        for model in models:
            avg_f1[model.name] = np.mean([instance['f1'][model.name] for instance in data if model.name in instance['f1']])
            avg_recall[model.name] = np.mean([instance['recall'][model.name] for instance in data if model.name in instance['recall']])
            avg_precision[model.name] = np.mean([instance['precision'][model.name] for instance in data if model.name in instance['precision']])
        results['avg_f1'] = avg_f1
        results['avg_recall'] = avg_recall
        results['avg_precision'] = avg_precision
        
    
    avg_num_labels = {}
    num_instances_labeled = {}
    for model in models:
        num_instances_labeled[model.name] = len([instance for instance in data if model.name in instance['sentence_types']])
        avg_num_labels[model.name] = \
            np.mean([len([t for t, p in instance['sentence_types'][model.name].items() if p >= model.accept_threshold])
                     for instance in data if model.name in instance['sentence_types']])
    if 'gold' in data[0]['sentence_types']:
        avg_num_labels['gold'] = np.mean([len(instance['sentence_types']['gold']) for instance in data])
    results['num_instances'] = len(data)
    results['num_sentence_types'] = len(SENTENCE_TYPES)
    results['avg_num_labels'] = avg_num_labels
    results['num_instances_labeled'] = num_instances_labeled
    results['thresholds'] = {model.name: model.accept_threshold for model in models}

    return results


def init_models():
    models_list = []
    if 'random' in MODELS_TO_USE:
        models_list.append(Model_random(temperature=MODELS_TO_USE['random']))
    if 'flan_t5_xxl' in MODELS_TO_USE:
        models_list.append(Model_flan_t5('flan_t5_xxl', temperature=MODELS_TO_USE['flan_t5_xxl']))
    if 'flan_t5_xl' in MODELS_TO_USE:
        models_list.append(Model_flan_t5('flan_t5_xl', temperature=MODELS_TO_USE['flan_t5_xl']))
    if 'flan_t5_large' in MODELS_TO_USE:
        models_list.append(Model_flan_t5('flan_t5_large', temperature=MODELS_TO_USE['flan_t5_large']))
    if 'flan_t5_base' in MODELS_TO_USE:
        models_list.append(Model_flan_t5('flan_t5_base', temperature=MODELS_TO_USE['flan_t5_base']))
    if 'flan_t5_small' in MODELS_TO_USE:
        models_list.append(Model_flan_t5('flan_t5_small', temperature=MODELS_TO_USE['flan_t5_small']))
    if 'flan_ul2' in MODELS_TO_USE:
        models_list.append(Model_flan_ul2(temperature=MODELS_TO_USE['flan_ul2']))
    if 'j2_jumbo_instruct' in MODELS_TO_USE:
        models_list.append(Model_j2_jumbo_instruct(temperature=MODELS_TO_USE['j2_jumbo_instruct']))
    return models_list

        
def main(input_gold_annotation_path, output_temp_intermediate_save_path, output_path):
    print('Loading dataset...')
    # data = get_test_set_data(input_gold_annotation_path, output_temp_intermediate_save_path)
    data = get_annotation_set_data(input_gold_annotation_path, output_temp_intermediate_save_path)
    models = init_models()
    failed_instances_in_models = {}
    elapsed_time_in_models = {}
    for model_idx, model in enumerate(models):
        print(f'Loading model {model_idx+1} of {len(models)}: {model.name}...')
        model.load()
        print(f'Inferring...')
        start_time = time.time()
        failed_instances = model.infer(data, output_temp_intermediate_save_path)
        end_time = time.time()
        failed_instances_in_models[model.name] = len(failed_instances)
        elapsed_time_in_models[model.name] = end_time - start_time
        print(f'Unloading model...')
        model.unload()
        print(f'Saving...')
        save_data(data, output_temp_intermediate_save_path)
        print(f'Done with {model.name}')
    
    print('Computing final results...')
    results = compute_final(models, data)
    results['failed_instances'] = failed_instances_in_models
    results['elapsed_time_total'] = elapsed_time_in_models
    
    print('Saving data and results...')
    save_data({'data': data, 'results': results}, output_path)
    print('Done')
    
    
if __name__ == '__main__':
    main(INPUT_GOLD_ANNOTATION_PATH, OUTPUT_TEMP_INTERMEDIATE_SAVE_PATH, OUTPUT_PATH)
