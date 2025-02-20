import random
import numpy as np
from Evaluation.EvaluationOnType import EvaluationOnType
from Utils.Utils import Utils, TYPES_ORDERED


class EvaluationOnTypeOpinions(EvaluationOnType):
    def __get_data_for_opinions_subset(self, input_file_path, split):
        data = Utils.get_data_predicted_types(input_file_path)

        for d in data:
            if 'opinion' not in d['sentence_types']['flan_t5_xxl']:
                d['sentence_types']['flan_t5_xxl']['opinion'] = 0.
            if 'opinion_with_reason' not in d['sentence_types']['flan_t5_xxl']:
                d['sentence_types']['flan_t5_xxl']['opinion_with_reason'] = 0.
            sent_types = d['sentence_types']['flan_t5_xxl']
            type_vector = [sent_types[t] if t in sent_types else 0. for t in TYPES_ORDERED]
            d['vector'] = type_vector

        data = [d for d in data if d['split'] == split]
        return data

    def __get_opinions_data(self, data_filepath):
        data_train = self.__get_data_for_opinions_subset(data_filepath, 'train')
        data_test = self.__get_data_for_opinions_subset(data_filepath, 'test')
        return data_train, data_test

    def __split_to_opinion_types(self, data):
        data_opinion = [d for d in data if d['is_opinion'] == 1]
        data_opinion_with_reason = [d for d in data if d['is_opinion_with_reason'] == 1]
        data_not_opinion = [d for d in data if d['is_opinion'] == 0]
        return data_opinion, data_opinion_with_reason, data_not_opinion

    def evaluate(self, input_file_path, train_set_size=None):
        data_train, data_test = self.__get_opinions_data(input_file_path)

        # get the opinion and opinion_with_reason thresholds from the train set:
        if len(data_train) > 0:
            data_opinion_train, data_opinion_with_reason_train, data_not_opinion_train = \
                self.__split_to_opinion_types(data_train)
            data_opinion_train_all = data_opinion_train + data_not_opinion_train
            data_opinion_with_reason_train_all = data_opinion_with_reason_train + data_not_opinion_train
            # if requested, take only a sample of the train set
            # (used for fining optimal threshold and not actual training):
            if train_set_size:
                print(f'Sampling train data with {train_set_size}')
                data_opinion_train_all = random.sample(data_opinion_train_all, train_set_size)
                data_opinion_with_reason_train_all = random.sample(data_opinion_with_reason_train_all, train_set_size)
            opt_threshold_opinion = \
                self.get_optimal_threshold_from_train_set(data_opinion_train_all, 'opinion')
            opt_threshold_opinion_with_reason = \
                self.get_optimal_threshold_from_train_set(data_opinion_with_reason_train_all,'opinion_with_reason')
        else:
            opt_threshold_opinion = -1
            opt_threshold_opinion_with_reason = -1

        # compute evaluation on test set:
        data_opinion_test, data_opinion_with_reason_test, data_not_opinion_test = \
            self.__split_to_opinion_types(data_test)
        acc_opinion, f1_opinion, prec_rec_vals_opinion, prec_pos_opinion = \
            self.compute_accuracy_on_test_set(data_opinion_test + data_not_opinion_test,
                                              'opinion', opt_threshold_opinion)
        acc_opinion_with_reason, f1_opinion_with_reason, prec_rec_vals_opinion_with_reason, prec_pos_opinion_with_reason = \
            self.compute_accuracy_on_test_set(data_opinion_with_reason_test + data_not_opinion_test,
                                              'opinion_with_reason', opt_threshold_opinion_with_reason)

        def print_results(name, opt_threshold, acc, f1, prec_rec_vals, num_positive, num_negative, train_size,
                          test_size):
            print(f'Results for {name}')
            print('---------------------')
            print(f'Num positive instances: {num_positive}')
            print(f'Num negative instances: {num_negative}')
            print(f'Train size: {train_size}')
            print(f'Test size: {test_size}')
            print(f'opt_threshold: {opt_threshold}')
            print(f'Accuracy: {acc}')
            print(f'F1: {f1}')
            for p in prec_rec_vals:
                print(f'Recall@Precision {p}: {prec_rec_vals[p]}')
            print()

        print_results('opinion', opt_threshold_opinion, acc_opinion,
                      f1_opinion, prec_rec_vals_opinion,
                      len(data_opinion_test), len(data_not_opinion_test),
                      len(data_opinion_train_all),
                      len(data_opinion_test + data_not_opinion_test))
        print_results('opinion_with_reason', opt_threshold_opinion_with_reason, acc_opinion_with_reason,
                      f1_opinion_with_reason, prec_rec_vals_opinion_with_reason,
                      len(data_opinion_with_reason_test), len(data_not_opinion_test),
                      len(data_opinion_with_reason_train_all),
                      len(data_opinion_with_reason_test + data_not_opinion_test))

        opinion_vec_avg = np.mean([d['vector'] for d in data_opinion_test], axis=0)
        opinion_with_reason_vec_avg = np.mean([d['vector'] for d in data_opinion_with_reason_test], axis=0)
        not_opinion_vec_avg = np.mean([d['vector'] for d in data_not_opinion_test], axis=0)
        Utils.show_vector_bar_plot({'opinion': opinion_vec_avg, 'not opinion': not_opinion_vec_avg})
        Utils.show_vector_bar_plot({'opinion with reason': opinion_with_reason_vec_avg, 'not opinion': not_opinion_vec_avg})
        Utils.show_vector_bar_plot({'opinion with reason': opinion_with_reason_vec_avg, 'opinion': opinion_vec_avg})
        Utils.show_vector_bar_plot({'opinion with reason': opinion_with_reason_vec_avg, 'opinion': opinion_vec_avg,
                                    'not opinion': not_opinion_vec_avg})


if __name__ == '__main__':
    input_file_path = '../data/type_predictions_opinions.json'
    train_set_size = 100  # None
    EvaluationOnTypeOpinions().evaluate(input_file_path, train_set_size)