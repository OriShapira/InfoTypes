import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import json

TYPES_ORDERED = ['opinion', 'opinion_with_reason', 'improvement_desire', 'comparative', 'comparative_general',
                 'buy_decision', 'speculative', 'personal_usage', 'situation', 'setup', 'tip', 'product_usage',
                 'product_description', 'price', 'compatibility', 'personal_info', 'general_info', 'comparative_seller',
                 'seller_experience', 'delivery_experience', 'imagery', 'sarcasm', 'rhetorical', 'inappropriate']

TYPE_GROUPS = {
    'opinion': ['opinion'],
    'opinion_with_reason': ['opinion_with_reason'],
    'opinions': ['opinion', 'opinion_with_reason'],
    'personal': ['personal_usage', 'personal_info'],
    'subjective': ['opinion', 'opinion_with_reason', 'improvement_desire', 'buy_decision',
                   'speculative', 'seller_experience', 'delivery_experience'],
    'objective': ['comparative', 'comparative_general', 'personal_usage', 'situation', 'setup', 'tip',
                  'product_usage', 'product_description', 'price', 'compatibility', 'general_info', 'comparative_seller'],
    'description': ['setup', 'tip', 'product_usage', 'product_description', 'price', 'compatibility'],
    'non_product': ['personal_info', 'general_info', 'comparative_seller', 'seller_experience', 'delivery_experience'],
    'linguistic': ['imagery', 'sarcasm', 'rhetorical', 'inappropriate'],
    'comparisons': ['comparative', 'comparative_general', 'comparative_seller'],
    'all': TYPES_ORDERED
}
TYPE_GROUP_ALL_ONLY = {'all': TYPES_ORDERED}

COARSE_TYPES_MAPPING = {
    'opinion': ['subjective', 'opinions'],
    'opinion_with_reason': ['subjective', 'opinions'],
    'improvement_desire': ['subjective'],
    'comparative': ['objective', 'comparisons'],
    'comparative_general': ['objective', 'comparisons'],
    'buy_decision': ['subjective'],
    'speculative': ['subjective'],
    'personal_usage': ['objective', 'personal'],
    'situation': ['objective'],
    'setup': ['objective', 'description'],
    'tip': ['objective', 'description'],
    'product_usage': ['objective', 'description'],
    'product_description': ['objective', 'description'],
    'price': ['objective', 'description'],
    'compatibility': ['objective', 'description'],
    'personal_info': ['personal', 'non_product'],
    'general_info': ['objective', 'non_product'],
    'comparative_seller': ['objective', 'comparisons', 'non_product'],
    'seller_experience': ['subjective', 'non_product'],
    'delivery_experience': ['subjective', 'non_product'],
    'imagery': ['stylistic'],
    'sarcasm': ['stylistic'],
    'rhetorical': ['stylistic'],
    'inappropriate': ['stylistic']
}

COARSE_TYPES_ORDERED = ['subjective', 'opinions', 'objective', 'description', 'comparisons', 'personal', 'non_product', 'stylistic']

# for converting labels to IDs in the huggingface model
ID_TO_LABEL_SENTIMENT = {0: "neg", 1: "pos"}
LABEL_TO_ID_SENTIMENT = {"neg": 0, "pos": 1}
ID_TO_LABEL_HELPFULNESS = {0: "unhelpful", 1: "helpful"}
LABEL_TO_ID_HELPFULNESS = {"unhelpful": 0, "helpful": 1}
# These two need to be set with one of the above two options before funning the classification:
ID_TO_LABEL = None
LABEL_TO_ID = None


class Utils:
    @staticmethod
    def get_data_predicted_types(input_file_path):
        with open(input_file_path) as fIn:
            data = json.load(fIn)
            # a tmp_data.json file has the data on the top level, a results.json file has it under 'data' key:
            if 'data' in data:
                data = data['data']
        # only use datums that have the scores computed for it:
        data = [d for d in data if 'flan_t5_xxl' in d['sentence_types']]
        return data

    @staticmethod
    def get_scores_of_types(type_to_score,
                            types_to_use_list,
                            use_coarse_grained_types=False,
                            mapped_type_aggregation_func=np.mean):

        if not use_coarse_grained_types:
            # create the scores vector in the order of the types_to_use_list
            type_vector = [type_to_score[t] if t in type_to_score else 0. for t in types_to_use_list]
        else:
            # get a list of scores for each mapped type:
            new_type_dict = {}  # mapped_type -> [scores of higher resolution types]
            for t in types_to_use_list:
                for mapped_type in COARSE_TYPES_MAPPING[t]:
                    if mapped_type not in new_type_dict:
                        new_type_dict[mapped_type] = []
                    new_type_dict[mapped_type].append(type_to_score[t] if t in type_to_score else 0.)
            # create the scores vector in the order of the mapped_type_to_use_list,
            # and use the aggregation function on the list
            type_vector = []
            for mapped_type in COARSE_TYPES_ORDERED:
                type_vector.append(mapped_type_aggregation_func(new_type_dict[mapped_type])
                                   if mapped_type in new_type_dict else 0.)
            # print('---')
            # print(type_to_score)
            # print(new_type_dict)
            # print(type_vector)
            # print('---')

        return type_vector

    @staticmethod
    def avg_list_of_results(results_list):
        avg_results = {}
        for result_name in results_list[0].keys():
            if result_name != 'predictions':
                values = [r[result_name] for r in results_list]
                if isinstance(values[0], Counter):
                    avg_result = dict(pd.DataFrame(values).mean())
                    avg_results[result_name] = avg_result
                elif isinstance(values[0], dict):
                    avg_results[result_name] = {}
                    p_vals = list(values[0].keys())
                    for p in p_vals:
                        cur_vals = [val[p] for val in values]
                        mean = np.nanmean(cur_vals)
                        lower, upper, alpha, ci = Utils.compute_confidence_intervals(cur_vals, mean=mean)
                        avg_results[result_name][p] = (mean, lower, upper, alpha, ci)
                    # avg_results[result_name] = {p: np.nanmean([val[p] for val in values]) for p in p_vals}
                else:
                    avg_result = np.mean(values, axis=0)
                    lower, upper, alpha, ci = Utils.compute_confidence_intervals(values, mean=avg_result)
                    avg_results[result_name] = (avg_result, lower, upper, alpha, ci)

        return avg_results

    @staticmethod
    def get_combined_X(X_vecs, y_true_vec):
        # group together the vectors for each class:
        X_vecs_per_class = {}
        for X, y in zip(X_vecs, y_true_vec):
            if y not in X_vecs_per_class:
                X_vecs_per_class[y] = []
            X_vecs_per_class[y].append(X)

        # average the vectors of each class:
        X_per_class = {}
        for y in X_vecs_per_class:
            X_per_class[y] = np.mean(X_vecs_per_class[y], axis=0)

        return X_per_class

    @staticmethod
    def show_vector_bar_plot(X_per_class):
        print(X_per_class)
        indexes = np.arange(len(TYPES_ORDERED))
        width = 1
        patterns = ["", "//", "..", 'xx']
        colors = [(30 / 255, 136 / 255, 229 / 255, 0.5), (216 / 255, 27 / 255, 96 / 255, 0.3),
                  (255 / 255, 193 / 255, 7 / 255, 0.3)]
        # colors = [(30/255,200/255,255/255,0.5), (216/255,27/255,96/255,0.3), (255/255,193/255,7/255,0.3)]

        # move "un", "not" or "neg" label to end of list:
        classes_info = list(X_per_class.items())
        neg_label_idx = -1
        for i, (y, X) in enumerate(classes_info):
            if y.startswith('un') or y.startswith('not') or y.startswith('neg'):
                neg_label_idx = i
        if neg_label_idx >= 0:
            data_len = len(classes_info)
            popped_item = classes_info.pop(neg_label_idx)
            classes_info.append(popped_item)

        # draw the plot:
        class_rects = []  # the rectangles of the bars in the plot for each class
        for i, (y, X) in enumerate(classes_info):
            rects = plt.bar(indexes, X, width * 0.8, label=y, edgecolor=(0, 0, 0, 1), hatch=patterns[i],
                            color=colors[i])  # alpha=0.3
            class_rects.append(rects)

        plt.xticks(indexes - 1 + width, TYPES_ORDERED, rotation='vertical')
        plt.legend()
        plt.title(f'Average sentence type')
        plt.show()


    @staticmethod
    def print_results_of_prediction_for_type_sets(types_set_to_results, model_class_name_of_results):

        def extract_val(res, metric):
            return res[metric][0] if isinstance(res[metric], tuple) else res[metric]

        # print comparison
        if model_class_name_of_results == 'SVMClassifier':
            print('--- Classification: F1_Macro ---')
            for types_set_name, results in types_set_to_results.items():
                print(f'{extract_val(results, "f1_macro") * 100:.1f} | {types_set_name}')
            print('--- Classification: Accuracy ---')
            for types_set_name, results in types_set_to_results.items():
                print(f'{extract_val(results, "accuracy") * 100:.1f} | {types_set_name}')
            print('--- Classification: F1 (on best class) ---')
            for types_set_name, results in types_set_to_results.items():
                print(f'{extract_val(results, "f1") * 100:.1f} | {types_set_name}')

        elif model_class_name_of_results == 'LinearRegression':
            print('--- Regression: MSE, Pearson, NDCG@1 ---')
            for types_set_name, results in types_set_to_results.items():
                print(f'{results["mean_squared_error"]:.3f} & {results["pearson"][0]:.2f} & {results["NDCG@1"]:.2f} | {types_set_name}')

    @staticmethod
    def compute_confidence_intervals(values, alpha=0.001, mean=None):
        # lower, upper = sms.DescrStatsW(cur_vals).tconfint_mean(alpha=alpha)

        try:
            values = [v for v in values if not np.isnan(v)]
            confidence = 1 - alpha
            choices = [np.random.choice(values, size=len(values), replace=True).mean() for i in range(1000)]
            lower, upper = np.percentile(choices, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)])
        except:
            lower, upper = sms.DescrStatsW(values).tconfint_mean(alpha=alpha)

        # get the max confidence interval
        if mean is not None:
            try:
                ci = max(upper - mean, mean - lower)
            except:
                ci = None
        else:
            ci = None

        return lower, upper, alpha, ci