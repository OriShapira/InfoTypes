class Prediction:
    def __init__(self, model_class, type_groups, use_coarse_grained_types):
        self.model_class = model_class
        self.type_groups = type_groups
        self.use_coarse_grained_types = use_coarse_grained_types
