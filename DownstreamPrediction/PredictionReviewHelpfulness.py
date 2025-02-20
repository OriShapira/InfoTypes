import sys
sys.path.append('..')

from DownstreamPrediction.PredictionReview import PredictionReview
from DownstreamPrediction.Models import MySVMClassifier, MyBERTClassifier
from Utils import Utils as Utils_consts


class PredictionReviewHelpfulness(PredictionReview):
    def __init__(self, input_file_path, model_class, type_groups, use_coarse_grained_types,
                 helpful_thresholds, nothelpful_thresholds):
        super().__init__(input_file_path, model_class, type_groups, use_coarse_grained_types)
        self.helpful_thresholds = helpful_thresholds
        self.nothelpful_thresholds = nothelpful_thresholds

    def get_datum_scores(self, review_info):
        helpful_count = review_info['helpful_count']
        nothelpful_count = review_info['nothelpful_count']

        helpful_count = helpful_count if helpful_count else 0
        nothelpful_count = nothelpful_count if nothelpful_count else 0
        # the proportion of helpful in all counts:
        instance_score = helpful_count / (helpful_count + nothelpful_count) if (helpful_count + nothelpful_count) > 0 else 0.5
        # set the class by the thresholds:
        if helpful_count >= self.helpful_thresholds[0] and nothelpful_count <= self.nothelpful_thresholds[1]:
            instance_class = 'helpful'
        elif helpful_count <= self.helpful_thresholds[0] and nothelpful_count >= self.nothelpful_thresholds[1]:
            instance_class = 'unhelpful'
        else:
            instance_class = 'neutral'

        return instance_score, instance_class

    def should_use_datum(self, gold_class):
        return gold_class in ['helpful', 'unhelpful']


if __name__ == '__main__':
    input_file_path = '../data/type_predictions_review_helpfulness.json'
    helpful_thresholds = (9, 0)
    nothelpful_thresholds = (0, 3)
    
    # SVM Classifier with types
    model_class = MySVMClassifier
    type_groups = Utils_consts.TYPE_GROUPS
    use_coarse_grained_types = False
    ## use coarse grained types
    # type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY
    # use_coarse_grained_types = True
    p = PredictionReviewHelpfulness(input_file_path, model_class, type_groups, use_coarse_grained_types,
                                    helpful_thresholds, nothelpful_thresholds)
    p.main()
    
    
    # RoBERTa Classifier with text
    Utils_consts.ID_TO_LABEL = Utils_consts.ID_TO_LABEL_HELPFULNESS  # for data tranformation for huggingface inference
    Utils_consts.LABEL_TO_ID = Utils_consts.LABEL_TO_ID_HELPFULNESS  # for data tranformation for huggingface inference
    model_class = MyBERTClassifier
    type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY  # not relevant for text (just run the training and prediction once)
    use_coarse_grained_types = False  # not relevant for text
    p = PredictionReviewHelpfulness(input_file_path, model_class, type_groups, use_coarse_grained_types,
                                    helpful_thresholds, nothelpful_thresholds)
    p.main()
