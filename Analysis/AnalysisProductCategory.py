import numpy as np
import matplotlib.pyplot as plt
from Utils.Utils import Utils, TYPES_ORDERED
from Analysis.AnalysisUtils import AnalysisUtils


class AnalysisProductCategory:

    def __show_vector_bar_plot_per_category(self, X_per_cat, types_to_show):
        x_Labels = types_to_show
        type_indices = [TYPES_ORDERED.index(t) for t in x_Labels]  # prepare the indices of the types to show
        indexes = np.arange(len(x_Labels))

        # percentage histograms on the category level:
        hatches = ['//', '\\\\', '||', '--', '+', 'x', 'o', 'O', '.', '*']
        width = 0.15
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        fig.set_figwidth(5.5)
        for i, (cat, vals) in enumerate(X_per_cat.items()):
            vals_to_show = [vals[t_idx] for t_idx in type_indices]
            offset = width * multiplier
            ax.bar(indexes + offset, vals_to_show, width, label=cat, edgecolor=(0,0,0,1), hatch=hatches[i], alpha=0.5)
            multiplier += 1

        ax.set_xticks(indexes + ((len(X_per_cat)-1)/2)*width, x_Labels, rotation=30, ha='right', rotation_mode="anchor")
        ax.set_title(f'Avg. sentence type vector per cateogry')
        ax.legend(loc='upper right', ncol=2)
        plt.show()

    def analyze(self, data_path, types_to_show, categories_to_show):

        data = Utils.get_data_predicted_types(data_path)

        # category -> doc_id -> [types_vecs of its sentences]
        cat_to_doc_id_to_type_vecs = {}
        for datum in data:
            # only look at the reviews (skip over the summaries):
            if 'review_id' in datum and not datum['review_id'].startswith('review'):
                continue

            cat = datum['category']
            review_id = datum['review_id'] if 'review_id' in datum else datum['sentence_idx']
            doc_id = AnalysisUtils.get_full_doc_id_used(datum['asin'], review_id)
            sent_types = datum['sentence_types']['flan_t5_xxl']
            types_vec = [sent_types[t] if t in sent_types else 0. for t in TYPES_ORDERED]

            # add the sentence vector to the current doc:
            if cat not in cat_to_doc_id_to_type_vecs:
                cat_to_doc_id_to_type_vecs[cat] = {}
            if doc_id not in cat_to_doc_id_to_type_vecs[cat]:
                cat_to_doc_id_to_type_vecs[cat][doc_id] = []
            cat_to_doc_id_to_type_vecs[cat][doc_id].append(types_vec)

        # get the review-level vectors per category:
        cat_to_vecs = {}  # category -> [review-level type vectors]
        for cat in cat_to_doc_id_to_type_vecs:
            for doc_id, type_vec_list in cat_to_doc_id_to_type_vecs[cat].items():
                if cat not in cat_to_vecs:
                    cat_to_vecs[cat] = []
                cat_to_vecs[cat].append(np.mean(type_vec_list, axis=0))

        # get the average vecs of the reviews of each subset:
        cat_to_vec = {}
        for cat, cat_vecs in cat_to_vecs.items():
            if categories_to_show and cat in categories_to_show:
                cat_to_vec[cat] = np.mean(cat_vecs, axis=0)

        # show results:
        for cat in cat_to_vecs:
            print(f'Name: {cat} | Num docs: {len(cat_to_vecs[cat])}')
        self.__show_vector_bar_plot_per_category(cat_to_vec, types_to_show)
        print('Vectors:')
        print(cat_to_vec)


if __name__ == '__main__':
    data_path = '../data/type_predictions_review_summarization.json'
    types_to_show = ['personal_usage', 'situation', 'setup', 'tip', 'personal_info']
    categories_to_show = ['Books', 'Electronics', 'Apparel', 'Toys and Games']
    types_to_display = ['personal_usage', 'situation', 'setup', 'tip', 'personal_info']
    AnalysisProductCategory().analyze(data_path, types_to_show, categories_to_show)