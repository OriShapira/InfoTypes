from DownstreamPrediction.Prediction import Prediction
from Utils.Utils import Utils
from collections import Counter
import numpy as np
import random
from tqdm import tqdm


class PredictionReview(Prediction):
    def __init__(self, input_file_path, model_class, type_groups, use_coarse_grained_types):
        super().__init__(model_class, type_groups, use_coarse_grained_types)
        self.input_file_path = input_file_path

    def get_review_data(self, specific_types_to_use):
        data = Utils.get_data_predicted_types(self.input_file_path)

        # get the review data, and aggregate it per review:
        revid_to_info = {}  # review_id -> {'helpful_count', 'nothelpful_count', 'sentences', 'type_vectors'}
        for datum in data:
            helpful_count = datum['review_helpful_count']
            nothelpful_count = datum['review_nothelpful_count']
            sent_types = datum['sentence_types']['flan_t5_xxl']
            sentence = datum['sentence']
            review_id = datum['review_id']
            rating = datum['review_overall_rating']

            if review_id not in revid_to_info:
                revid_to_info[review_id] = {'helpful_count': helpful_count,
                                            'nothelpful_count': nothelpful_count,
                                            'rating': rating,
                                            'sentences': [],
                                            'type_vectors': []}

            revid_to_info[review_id]['sentences'].append(sentence)
            type_vector = Utils.get_scores_of_types(sent_types, specific_types_to_use,
                                                    use_coarse_grained_types=self.use_coarse_grained_types)
            revid_to_info[review_id]['type_vectors'].append(type_vector)

        # compute the average sent_type scores in each review for the full data, and create the new data list:
        full_data = []
        class_count = Counter()
        for review_id, review_info in revid_to_info.items():
            gold_score, gold_class = self.get_datum_scores(review_info)

            # the new datum type:
            if self.should_use_datum(gold_class):
                full_data.append({'text_id': review_id,
                                  'text': ' '.join(review_info['sentences']),
                                  'gold_score': gold_score,
                                  'gold_class': gold_class,
                                  'vector': np.mean(review_info['type_vectors'], axis=0)})
                class_count[gold_class] += 1

        return full_data, class_count

    def get_results_on_data(self, specific_types_to_use):
        print('Getting Data...')
        data, class_count = self.get_review_data(specific_types_to_use)

        all_results = []
        for i in tqdm(range(50)):
            # split to train/test:
            random.shuffle(data)  # shuffle the balanced data
            train_size = int(0.7 * len(data))  # train set size is 50%
            train_set, test_set = data[:train_size], data[train_size:]  # split

            # train:
            model = self.model_class()
            model.train_model(train_set)

            # test:
            model.predict_and_score(test_set)
            all_results.append(model.results)

        # average the results over all train/test runs:
        avg_results = Utils.avg_list_of_results(all_results)
        last_X_vecs = model.X_vecs
        last_y_true_vec = model.y_true_vec

        # force the overall results and print:
        model = self.model_class()
        model.results = avg_results
        model.X_vecs = last_X_vecs
        model.y_true_vec = last_y_true_vec
        model.show_results()
        return avg_results

    def get_datum_scores(self, review_info):
        raise NotImplementedError()

    def should_use_datum(self, gold_class):
        raise NotImplementedError()

    def main(self):
        types_set_to_results = {}
        for types_set_name, types_set in self.type_groups.items():
            print(f'Type set: {types_set_name}:')
            print('-----------------------')
            types_set_to_results[types_set_name] = self.get_results_on_data(specific_types_to_use=types_set)
            print('\n\n')
        Utils.print_results_of_prediction_for_type_sets(types_set_to_results, self.model_class().model_name)
