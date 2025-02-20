from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from Utils.Utils import Utils, TYPES_ORDERED
from AnalysisUtils import AnalysisUtils


class AnalysisRhetoricalStructure:
    def analyze(self, data_path, doc_type, sent_len_to_analyze, types_to_show):
        # doc_type is 'review' or 'summary'

        data = Utils.get_data_predicted_types(data_path)

        # get doc_id -> num_sentence
        doc_id_to_sentence_count = Counter()
        for datum in data:
            asin = datum['asin']
            doc_id = str(datum['review_id'])

            # skip docs not of the requested type
            if not doc_id.startswith(doc_type):
                continue

            # count the sentence for the doc:
            doc_id = AnalysisUtils.get_full_doc_id_used(asin, doc_id)
            doc_id_to_sentence_count[doc_id] += 1

        print(f'Avg. num sentences per doc: {np.mean(list(doc_id_to_sentence_count.values()))}')
        print(f'Med. num sentences per doc: {np.median(list(doc_id_to_sentence_count.values()))}')

        # get the doc IDs to use for this analysis (with the proper number of sentences):
        doc_ids_to_use = set()
        for doc_id, doc_len in doc_id_to_sentence_count.items():
            if doc_len == sent_len_to_analyze:
                doc_ids_to_use.add(doc_id)

        # get the types at each sentence index:
        sent_idx_to_type_dicts = {i: [] for i in range(sent_len_to_analyze)}  # sent_idx -> list of dicts of type_to_score
        doc_id_to_sent_count = Counter()
        for datum in data:
            asin = datum['asin']
            review_id = AnalysisUtils.get_doc_id_used(str(datum['review_id']))
            doc_id = AnalysisUtils.get_full_doc_id_used(asin, review_id)
            if doc_id in doc_ids_to_use:
                sentence_idx = doc_id_to_sent_count[doc_id]  # datum['sentence_idx']
                sent_types = datum['sentence_types']['flan_t5_xxl']
                types_vals = {t: sent_types[t] if t in sent_types else 0. for t in TYPES_ORDERED}
                sent_idx_to_type_dicts[sentence_idx].append(types_vals)
                doc_id_to_sent_count[doc_id] += 1

        # get the average for each type at each sentence index:
        type_to_sent_idx_to_avg_score = {}  # type -> sent_idx -> score
        num_docs = len(sent_idx_to_type_dicts[0])
        print(f'Num reviews with sentence count {sent_len_to_analyze}: {num_docs}')
        if not types_to_show:
            types_to_show = TYPES_ORDERED
        for sent_type in types_to_show:
            type_to_sent_idx_to_avg_score[sent_type] = {}
            for sent_idx in range(sent_len_to_analyze):
                avg_score = np.nanmean([sent_idx_to_type_dicts[sent_idx][i][sent_type] for i in range(num_docs)])
                type_to_sent_idx_to_avg_score[sent_type][sent_idx] = avg_score

        # show a line graph of the types over the sentence idx:
        plt.figure(figsize=(5, 5))
        x_vals = [i + 1 for i in range(sent_len_to_analyze)]
        markers = ['.', 'o', 'x', 'd', '*', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+',
                   'x', 'D', 'd']
        for line_idx, sent_type in enumerate(type_to_sent_idx_to_avg_score):
            y_vals = [type_to_sent_idx_to_avg_score[sent_type][i] for i in range(sent_len_to_analyze)]
            if self.is_not_straight_line(y_vals):
                plt.plot(x_vals, y_vals, label=sent_type, marker=markers[line_idx])

        if doc_type == 'review':
            plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.63))  # for amasum reviews
        elif doc_type == 'summary':
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # for amasum summaries

        plt.title('Document Structure')
        plt.xlabel('Sentence Index')
        plt.ylabel('Avg. Probability of Type')
        plt.show()

    def is_not_straight_line(self, vals):
        abs_diffs = [abs(t - s) for s, t in zip(vals, vals[1:])]
        sum_abs_diffs = sum(abs_diffs)
        return sum_abs_diffs > 0.1


if __name__ == '__main__':
    data_path = '../data/type_predictions_review_summarization.json'

    types_to_show = ['opinion', 'opinion_with_reason', 'buy_decision', 'speculative', 'product_description', 'tip', 'personal_usage']
    sent_len_to_analyze = 6
    print('------- REVIEWS ---------')
    AnalysisRhetoricalStructure().analyze(data_path, 'review', sent_len_to_analyze, types_to_show)

    types_to_show = ['opinion_with_reason', 'speculative', 'product_description', 'tip', 'buy_decision', 'improvement_desire', 'speculative']
    sent_len_to_analyze = 7
    print('------- SUMMARIES ---------')
    AnalysisRhetoricalStructure().analyze(data_path, 'summary', sent_len_to_analyze, types_to_show)
