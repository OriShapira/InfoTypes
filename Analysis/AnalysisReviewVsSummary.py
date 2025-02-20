import numpy as np
import matplotlib.pyplot as plt
from Utils.Utils import Utils, TYPES_ORDERED
from AnalysisUtils import AnalysisUtils


class AnalysisReviewVsCategory:

    def __show_vector_bar_plot(self, X_per_class):
        indexes = np.arange(len(TYPES_ORDERED))
        width = 1
        patterns = ["", "//", 'xx']
        colors = [(0 / 255, 153 / 255, 0 / 255, 0.5), (255 / 255, 128 / 255, 0 / 255, 0.4),
                  (255 / 255, 0 / 255, 127 / 255, 0.3), (102 / 255, 204 / 255, 0 / 255, 0.3),
                  (255 / 255, 193 / 255, 7 / 255, 0.3), (153 / 255, 0 / 255, 153 / 255, 0.3)]

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

    def analyze(self, data_path):
        data = Utils.get_data_predicted_types(data_path)

        def is_review(review_id):
            return review_id.startswith('review')

        def is_summary(review_id):
            return review_id.startswith('summary')

        # subset -> doc_id -> [types_vecs of its sentences]
        subset_to_doc_id_to_type_vecs = {}
        for datum in data:
            if 'flan_t5_xxl' not in datum['sentence_types']:
                continue

            for is_subset_func, subset_name in [(is_review, 'reviews'), (is_summary, 'summaries')]:
                if is_subset_func(datum['review_id']):

                    doc_id = AnalysisUtils.get_full_doc_id_used(datum['asin'], datum['review_id'])
                    sent_types = datum['sentence_types']['flan_t5_xxl']
                    types_vec = [sent_types[t] if t in sent_types else 0. for t in TYPES_ORDERED]

                    # add the sentence vector to the current doc:
                    if subset_name not in subset_to_doc_id_to_type_vecs:
                        subset_to_doc_id_to_type_vecs[subset_name] = {}
                    if doc_id not in subset_to_doc_id_to_type_vecs[subset_name]:
                        subset_to_doc_id_to_type_vecs[subset_name][doc_id] = []
                    subset_to_doc_id_to_type_vecs[subset_name][doc_id].append(types_vec)

        # get the review-level vectors per subset:
        subset_to_vecs = {}  # subset_name -> [review-level type vectors]
        for subset_name in subset_to_doc_id_to_type_vecs:
            for doc_id, type_vec_list in subset_to_doc_id_to_type_vecs[subset_name].items():
                if subset_name not in subset_to_vecs:
                    subset_to_vecs[subset_name] = []
                subset_to_vecs[subset_name].append(np.mean(type_vec_list, axis=0))

        # get the average vecs of the reviews of each subset:
        subset_to_vec = {}
        for subset_name, subset_vecs in subset_to_vecs.items():
            subset_to_vec[subset_name] = np.mean(subset_vecs, axis=0)

        # show results:
        for subset_name in subset_to_vecs:
            print(f'Name: {subset_name} | Num docs: {len(subset_to_vecs[subset_name])}')
        self.__show_vector_bar_plot(subset_to_vec)
        print('Vectors:')
        print(subset_to_vec)


if __name__ == '__main__':
    data_path = '../data/type_predictions_review_summarization.json'
    AnalysisReviewVsCategory().analyze(data_path)