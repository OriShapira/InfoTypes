from Utils.Utils import Utils, TYPES_ORDERED

class ThresholdOptimization:

    @staticmethod
    def __sent_type_in_data(data, sent_type):
        # return True the first time the type is found
        for instance in data:
            if sent_type in instance['sentence_types']['gold']:
                return True
            if sent_type in instance['sentence_types']['flan_t5_xxl']:
                return True
        return False

    @staticmethod
    def find_optimal_thresholds(data, sentence_types):
        # Find the optimal threshold for each of the types.
        # Compute the F1 score for each type over all instances.
        # I.e., for TP and FP, a "positive" is when a label is chosen by the model and a "negative" is when the
        # model doesn't choose the label.
        possible_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        DEFAULT_THRESHOLD = 0.5  # if a type is not found at all, use this value

        type_to_threshold_to_f1 = {ty: {th: -1 for th in possible_thresholds} for ty in sentence_types}
        for sent_type in sentence_types:
            # if the type does not appear in the data, mark the default threshold with a high score:
            if not ThresholdOptimization.__sent_type_in_data(data, sent_type):
                type_to_threshold_to_f1[sent_type][DEFAULT_THRESHOLD] = 1
            # otherwise, search for the optimal threshold:
            else:
                for threshold in possible_thresholds:
                    tp = 0
                    fp = 0
                    fn = 0
                    for instance in data:
                        labels_gold = instance['sentence_types']['gold']
                        labels_pred = set([t for t, p in instance['sentence_types']['flan_t5_xxl'].items() if p >= threshold])
                        if sent_type in labels_gold and sent_type in labels_pred:
                            tp += 1
                        elif sent_type in labels_gold and sent_type not in labels_pred:
                            fn += 1
                        elif sent_type not in labels_gold and sent_type in labels_pred:
                            fp += 1

                    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.
                    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.
                    type_to_threshold_to_f1[sent_type][threshold] = f1

        type_to_opt_threshold = {}
        for sent_type in type_to_threshold_to_f1:
            best_threshold = max(type_to_threshold_to_f1[sent_type], key=type_to_threshold_to_f1[sent_type].get)
            type_to_opt_threshold[sent_type] = best_threshold

        return type_to_opt_threshold


if __name__ == '__main__':
    data = Utils.get_data_predicted_types('../data/dev_set.json')
    sentence_types = TYPES_ORDERED
    type_to_opt_threshold = ThresholdOptimization.find_optimal_thresholds(data, sentence_types)
    print(type_to_opt_threshold)
