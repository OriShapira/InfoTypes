from tqdm import tqdm
import random
import numpy as np
from Evaluation.EvaluationOnType import EvaluationOnType
from Utils.Utils import Utils, TYPES_ORDERED


class EvaluationOnTypeTip(EvaluationOnType):

    def __get_data_for_tips(self, input_file_path, remove_warnings):
        data = Utils.get_data_predicted_types(input_file_path)

        for d in data:
            if 'tip' not in d['sentence_types']['flan_t5_xxl']:
                d['sentence_types']['flan_t5_xxl']['tip'] = 0.
            d['gold_class'] = d['is_tip']

            # the sentence type vector (to show how tips an non-tips differ on all types):
            sent_types = d['sentence_types']['flan_t5_xxl']
            type_vector = [sent_types[t] if t in sent_types else 0. for t in TYPES_ORDERED]
            d['vector'] = type_vector

        data_tips = [d for d in data if d['is_tip'] == '1' and (
                not remove_warnings or d['tip_type'] != "Warning")]  # removing the Warning types
        data_not_tips = [d for d in data if d['is_tip'] == '0']

        print(f'Num tips total: {len(data_tips)}')
        print(f'Num non-tips total: {len(data_not_tips)}')

        return data_tips, data_not_tips

    def __split_data(self, data_tips, data_not_tips, force_train_size):
        # split to train/test:
        dev_set = data_tips + random.sample(data_not_tips, len(data_tips))  # balanced data (tips-to-nottips)
        random.shuffle(dev_set)  # shuffle the balanced data
        train_size = int(0.8 * len(dev_set))  # train set size is 80%
        train_set, test_set = dev_set[:train_size], dev_set[train_size:]  # split
        # the train set size can be set with a parameter:
        if force_train_size:
            train_set = random.sample(train_set, force_train_size)
        return train_set, test_set

    def evaluate(self, input_file_path, remove_warnings=False, num_cross_val_repeats=50, force_train_size=None):

        data_tips, data_not_tips = self.__get_data_for_tips(input_file_path, remove_warnings)

        # compute the
        all_standard_results = []
        for i in tqdm(range(num_cross_val_repeats)):
            # get data splits:
            train_set, test_set = self.__split_data(data_tips, data_not_tips, force_train_size)

            # get the optimal tip threshold to maximize accuracy (nothing is trained really, we just use this data
            # to find the optimal threshold for the type):
            opt_threshold = self.get_optimal_threshold_from_train_set(train_set, 'tip') if len(train_set) > 0 else 0.5

            # compute evaluation on test set:
            acc, f1, prec_rec_vals, prec_pos = self.compute_accuracy_on_test_set(test_set, 'tip', opt_threshold)
            all_standard_results.append({'acc': acc, 'f1': f1, 'prec_rec_vals': prec_rec_vals,
                                         'prec_pos': prec_pos, 'opt_thresh': opt_threshold})

        # average the results over all train/test runs:
        avg_standard_results = Utils.avg_list_of_results(all_standard_results)

        # show average results over the sample runs:
        print(f'avg_opt_threshold: {avg_standard_results["opt_thresh"]}')
        print(f'Accuracy: {avg_standard_results["acc"]}')
        print(f'F1: {avg_standard_results["f1"]}')
        print(f'Precision on positive: {avg_standard_results["prec_pos"]}')
        for p in avg_standard_results["prec_rec_vals"]:
            print(f'Recall@Precision {p}: {avg_standard_results["prec_rec_vals"][p]}')
            print(
                f'Recall@Precision {p} count: {np.count_nonzero(~np.isnan([res["prec_rec_vals"][p] for res in all_standard_results]))}')

        not_tips_vec_avg = np.mean([d['vector'] for d in data_not_tips], axis=0)
        tips_vec_avg = np.mean([d['vector'] for d in data_tips], axis=0)
        Utils.show_vector_bar_plot({'tip': tips_vec_avg, 'not tip': not_tips_vec_avg})


if __name__ == '__main__':
    input_file_path = '../data/type_predictions_tips.json'
    remove_warning_tips = True
    num_cross_val_repeats = 50
    force_train_size = 100  # None

    EvaluationOnTypeTip().evaluate(input_file_path,
                                   remove_warnings=remove_warning_tips,
                                   num_cross_val_repeats=num_cross_val_repeats,
                                   force_train_size=force_train_size)
