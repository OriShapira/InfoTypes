import json
from collections import Counter
import numpy as np
from Utils.Utils import Utils, COARSE_TYPES_MAPPING, TYPES_ORDERED, COARSE_TYPES_ORDERED
from Evaluation.ThresholdOptimization import ThresholdOptimization


class EvaluationFull:

    def __get_data(self, input_file_path):
        with open(input_file_path) as fIn:
            data = json.load(fIn)['data']
        for d in data:
            d['instance_id'] = f'{d["asin"]}_{d["review_id"]}_{d["sentence_idx"]}'
        return data

    def __get_types_per_sentence(self, data_gold, data_pred, type_to_threshold, use_coarse_grained_types):
        types_per_sentence = {}  # instance_id -> {'gold': [<types>], 'pred': [<types>]}
        type_counter_total = {'gold': Counter(), 'pred': Counter()}  # 'gold'|'pred' -> type -> count_total
        for d_gold, d_pred in zip(data_gold, data_pred):
            assert(d_gold["instance_id"] == d_pred["instance_id"])
            instance_id = d_gold["instance_id"]
            types_per_sentence[instance_id] = {}
            # get the gold datum types:
            d_types_gold = [t for t in d_gold["sentence_types"]["gold"] if t != '']
            types_per_sentence[instance_id]['gold'] = d_types_gold if not use_coarse_grained_types \
                else self.__map_to_new_types(d_types_gold, COARSE_TYPES_MAPPING)
            type_counter_total['gold'].update(types_per_sentence[instance_id]['gold'])
            # get the pred datum types:
            d_types_pred = [t for t, p in d_pred["sentence_types"]["flan_t5_xxl"].items() if p >= type_to_threshold[t]]
            types_per_sentence[instance_id]['pred'] = d_types_pred if not use_coarse_grained_types \
                else self.__map_to_new_types(d_types_pred, COARSE_TYPES_MAPPING)
            type_counter_total['pred'].update(types_per_sentence[instance_id]['pred'])

        return types_per_sentence, type_counter_total

    def __compute_f1_scores(self, types_per_sentence, use_coarse_grained_types):
        types_to_use = TYPES_ORDERED if not use_coarse_grained_types else COARSE_TYPES_ORDERED
        type_to_f1 = {}
        type_to_rec = {}
        type_to_prec = {}
        for sent_type in types_to_use:
            tp = 0
            fp = 0
            fn = 0
            for instance_id, instance_results in types_per_sentence.items():
                labels_gold = instance_results['gold']
                labels_pred = instance_results['pred']
                if sent_type in labels_gold and sent_type in labels_pred:
                    tp += 1
                elif sent_type in labels_gold and sent_type not in labels_pred:
                    fn += 1
                elif sent_type not in labels_gold and sent_type in labels_pred:
                    fp += 1

            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.
            f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.
            type_to_f1[sent_type] = f1
            type_to_rec[sent_type] = recall
            type_to_prec[sent_type] = precision

        return type_to_f1, type_to_rec, type_to_prec

    def evaluate(self, input_gold_filepath, input_pred_filepath, type_to_threshold, use_coarse_grained_types):
        # get the raw data:
        data_gold = self.__get_data(input_gold_filepath)
        data_pred = self.__get_data(input_pred_filepath)

        # get the gold and pred types for each sentence:
        types_per_sentence, type_counter_total = self.__get_types_per_sentence(data_gold, data_pred,
                                                                               type_to_threshold,
                                                                               use_coarse_grained_types)

        # compute the F1 score for each of the sentence types:
        type_to_f1, type_to_rec, type_to_prec = self.__compute_f1_scores(types_per_sentence, use_coarse_grained_types)

        # print out counts, F1s and average F1 (macro-F1):
        num_sents_total = len(types_per_sentence)
        macro_f1 = np.mean(list(type_to_f1.values()))
        macro_rec = np.mean(list(type_to_rec.values()))
        macro_prec = np.mean(list(type_to_prec.values()))
        print('Type & F1 & Recall & Precision & \# (%) of Gold & \# (%) of Pred \\\\')
        types_ordered = TYPES_ORDERED if not use_coarse_grained_types else COARSE_TYPES_ORDERED
        for sent_type in types_ordered:
            sent_type_str = sent_type.replace("_", "\_")
            out_str = ''
            out_str += f'{sent_type_str} & '
            out_str += f'{type_to_f1[sent_type] * 100:.1f} & '
            out_str += f'{type_to_rec[sent_type] * 100:.1f} & '
            out_str += f'{type_to_prec[sent_type] * 100:.1f} & '
            out_str += f'{type_counter_total["gold"][sent_type]} '
            out_str += f'({type_counter_total["gold"][sent_type] / num_sents_total * 100:.1f}) & '
            out_str += f'{type_counter_total["pred"][sent_type]} '
            out_str += f'({type_counter_total["pred"][sent_type] / num_sents_total * 100:.1f}) \\\\'
            print(out_str)
        print('\midrule')
        print(f'ALL (Avg.) & {macro_f1 * 100:.1f} & {macro_rec * 100:.1f} & {macro_prec * 100:.1f} & - & - \\\\')

    def __map_to_new_types(self, orig_types_list, new_type_mapping):
        # create a list of new types, mapped from the given mapping:
        new_types_list = []
        for t in orig_types_list:
            new_types_list.extend(new_type_mapping[t])
        new_types_list = list(set(new_types_list))
        return new_types_list


if __name__ == '__main__':
    dev_set_filepath = '../data/dev_set.json'
    data = Utils.get_data_predicted_types(dev_set_filepath)
    sentence_types = TYPES_ORDERED
    type_to_opt_threshold = ThresholdOptimization.find_optimal_thresholds(data, sentence_types)

    test_set_gold_filepath = '../data/test_set_gold.json'
    test_set_pred_filepath = '../data/test_set_pred.json'
    use_coarse_grained_types = False
    EvaluationFull().evaluate(test_set_gold_filepath, test_set_pred_filepath,
                              type_to_opt_threshold, use_coarse_grained_types)
    print()
    print()
    print('COARSE-GRAINED:')
    use_coarse_grained_types = True
    EvaluationFull().evaluate(test_set_gold_filepath, test_set_pred_filepath,
                              type_to_opt_threshold, use_coarse_grained_types)
