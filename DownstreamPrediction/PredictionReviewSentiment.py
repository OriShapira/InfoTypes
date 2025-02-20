import sys
sys.path.append('..')

from DownstreamPrediction.PredictionReview import PredictionReview
from DownstreamPrediction.Models import MySVMClassifier, MyBERTClassifier
from Utils import Utils as Utils_consts


class PredictionReviewSentiment(PredictionReview):
    def __init__(self, input_file_path, model_class, type_groups, use_coarse_grained_types, use_full_rating):
        super().__init__(input_file_path, model_class, type_groups, use_coarse_grained_types)
        self.use_full_rating = use_full_rating

    def get_datum_scores(self, review_info):
        review_rating = review_info['rating']
        instance_score = review_rating
        instance_class = review_rating if self.use_full_rating else ('pos' if review_rating >= 4 else 'neg')
        return instance_score, instance_class

    def should_use_datum(self, gold_class):
        return True


if __name__ == '__main__':
    input_file_path = '../data/type_predictions_review_sentiment.json'
    
    # SVM Classifier with types
    model_class = MySVMClassifier
    type_groups = Utils_consts.TYPE_GROUPS
    use_coarse_grained_types = False
    ## use coarse grained types
    #type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY
    #use_coarse_grained_types = True
    use_full_rating = False  # False for pos/neg or True for 1-5
    p = PredictionReviewSentiment(input_file_path, model_class, type_groups, use_coarse_grained_types, use_full_rating)
    p.main()
    
    # RoBERTa Classifier with text
    Utils_consts.ID_TO_LABEL = Utils_consts.ID_TO_LABEL_SENTIMENT  # for data tranformation for huggingface inference
    Utils_consts.LABEL_TO_ID = Utils_consts.LABEL_TO_ID_SENTIMENT  # for data tranformation for huggingface inference
    model_class = MyBERTClassifier
    type_groups = Utils_consts.TYPE_GROUP_ALL_ONLY  # not relevant for text (just run the training and prediction once)
    use_coarse_grained_types = False  # not relevant for text
    use_full_rating = False  # False for pos/neg or True for 1-5
    p = PredictionReviewSentiment(input_file_path, model_class, type_groups, use_coarse_grained_types, use_full_rating)
    p.main()
