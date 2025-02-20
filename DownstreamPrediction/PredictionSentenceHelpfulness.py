import sys
sys.path.append('..')

from DownstreamPrediction.Prediction import Prediction
from DownstreamPrediction.Models import MySVMClassifier, MyLinearRegression, MyBERTClassifier
from Utils import Utils as Utils_consts
from Utils.Utils import Utils
from collections import Counter


class PredictionSentenceHelpfulness(Prediction):
    def __init__(self, input_file_path_train, input_file_path_test, model_class, type_groups, use_coarse_grained_types,
                 use_neutral_class):
        super().__init__(model_class, type_groups, use_coarse_grained_types)
        self.input_file_path_train = input_file_path_train
        self.input_file_path_test = input_file_path_test
        self.use_neutral_class = use_neutral_class

    def get_data_subset(self, input_file_path, specific_types_to_use, hs_score_lower=None, hs_score_upper=None):
        data = Utils.get_data_predicted_types(input_file_path)

        # get the HS data:
        full_data = []
        all_helpful_scores = []
        for datum in data:
            helpful_score = datum['helpful_score']
            sent_types = datum['sentence_types']['flan_t5_xxl']
            sentence = datum['sentence']
            sentence_idx = datum['sentence_idx']

            all_helpful_scores.append(helpful_score)

            # the sentence type vector:
            type_vector = Utils.get_scores_of_types(sent_types, specific_types_to_use,
                                                    use_coarse_grained_types=self.use_coarse_grained_types)

            # the new datum type:
            full_data.append({'text_id': sentence_idx,
                              'text': sentence,
                              'gold_score': helpful_score,
                              'gold_class': None,
                              'vector': type_vector})

        # get the top third and lower third score (if not given as an argument - in case of train set):
        if hs_score_lower is None or hs_score_upper is None:
            all_helpful_scores.sort()
            hs_score_lower = all_helpful_scores[int(len(all_helpful_scores) / 3)]
            hs_score_upper = all_helpful_scores[int(2 * len(all_helpful_scores) / 3)]

        # set the three classes by scores of all the HSs:
        class_counter = Counter()
        for datum in full_data:
            if datum['gold_score'] >= hs_score_upper:
                datum['gold_class'] = 'helpful'
            elif datum['gold_score'] <= hs_score_lower:
                datum['gold_class'] = 'unhelpful'
            else:
                datum['gold_class'] = 'neutral'
            class_counter[datum['gold_class']] += 1

        # might need to remove the neutral instances:
        if not self.use_neutral_class:
            new_data = [d for d in full_data if d['gold_class'] != 'neutral']
        else:
            new_data = full_data

        print(f'score lower: {hs_score_lower}')
        print(f'score upper: {hs_score_upper}')
        print(f'Class count: {class_counter}')
        print(f'Using neutral: {use_neutral_class}')

        return new_data, hs_score_lower, hs_score_upper

    def get_hs_data(self, specific_types_to_use):
        data_train, hs_score_lower, hs_score_upper = self.get_data_subset(self.input_file_path_train,
                                                                          specific_types_to_use)
        data_test, _, _ = self.get_data_subset(self.input_file_path_test, specific_types_to_use,
                                               hs_score_lower=hs_score_lower, hs_score_upper=hs_score_upper)
        return data_train, data_test

    def get_results_on_data(self, specific_types_to_use):
        print('Getting Data...')
        data_train, data_test = self.get_hs_data(specific_types_to_use)
        print('Training Model...')
        model = self.model_class()
        model.train_model(data_train)
        print('Testing Model...')
        model.predict_and_score(data_test)
        model.show_results()

        return model.results

    def main(self):
        types_set_to_results = {}
        for types_set_name, types_set in self.type_groups.items():
            print(f'Type set: {types_set_name}:')
            print('-----------------------')
            types_set_to_results[types_set_name] = self.get_results_on_data(specific_types_to_use=types_set)
            print('\n\n')
        Utils.print_results_of_prediction_for_type_sets(types_set_to_results, self.model_class().model_name)


if __name__ == '__main__':
    input_file_path_train = '../data/type_predictions_sentence_helpfulness_train.json'
    input_file_path_test = '../data/type_predictions_sentence_helpfulness_test.json'
    
    # Linear Regression with types
    model_class = MyLinearRegression
    type_groups = Utils_consts.TYPE_GROUPS
    use_coarse_grained_types = False
    ## use coarse grained types
    #type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY
    #use_coarse_grained_types = True
    use_neutral_class = True  # not relevant for regression
    PredictionSentenceHelpfulness(input_file_path_train, input_file_path_test, model_class, type_groups,
                                  use_coarse_grained_types, use_neutral_class).main()

    # SVM Classifier with types
    model_class = MySVMClassifier
    type_groups = Utils_consts.TYPE_GROUPS
    use_coarse_grained_types = False
    ## use coarse grained types
    #type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY
    #use_coarse_grained_types = True
    use_neutral_class = False  # only use "helpful" and "unhelpful" (and not the "neutral" class)
    PredictionSentenceHelpfulness(input_file_path_train, input_file_path_test, model_class, type_groups,
                                  use_coarse_grained_types, use_neutral_class).main()

    
    # RoBERTa Classifier with text
    Utils_consts.ID_TO_LABEL = Utils_consts.ID_TO_LABEL_HELPFULNESS  # for data tranformation for huggingface inference
    Utils_consts.LABEL_TO_ID = Utils_consts.LABEL_TO_ID_HELPFULNESS  # for data tranformation for huggingface inference
    model_class = MyBERTClassifier
    type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY  # not relevant for text (just run the training and prediction once)
    use_coarse_grained_types = False  # not relevant for text
    use_neutral_class = False  # not relevant for text
    PredictionSentenceHelpfulness(input_file_path_train, input_file_path_test, model_class, type_groups,
                                  use_coarse_grained_types, use_neutral_class).main()

    