from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
import numpy as np
from collections import Counter
from Utils import Utils as Utils_consts
from Utils.Utils import Utils

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import DatasetDict, Dataset
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments, Trainer
import torch


class Model:
    def __init__(self):
        self.model_name = 'Base'
        self.score_type = ''  # score_type is 'gold_score' or 'gold_class'
        self.input_type = ''  # input_type is 'vector' or 'text'
        self.results = {}

    def train_model(self, data_train):
        X, y = self.get_x_y_lists(data_train)
        model_instance = self._model_init()
        self.model = self._model_fit(model_instance, X, y)
        self.results = {}

    def _model_init(self):
        pass
        
    def _model_fit(self, model_intialized, X, y):
        pass
    
    def _model_predict_and_score(self, X, y):
        pass

    def predict_and_score(self, data_test):
        pass

    def show_results(self):
        print('---')
        print(f'Model results: {self.model_name}')
        print('---')
        for result_name, result_val in self.results.items():
            if result_name != 'predictions':
                if result_name == 'confusion_matrix':
                    print(f'{result_name}:\n{result_val}')
                else:
                    print(f'{result_name}: {result_val}')

    def get_x_y_lists(self, data):
        X = []
        y = []
        for datum in data:
            y.append(datum[self.score_type])
            X.append(datum[self.input_type])
        return np.array(X), np.array(y)


class MySVMClassifier(Model):
    def __init__(self):
        super()
        self.model_name = 'SVMClassifier'
        self.score_type = 'gold_class'
        self.input_type = 'vector'

    def _model_init(self):
        return SVC(kernel='linear', C=1)
        
    def _model_fit(self, model_intialized, X, y):
        return model_intialized.fit(X, y)
    
    def _model_predict_and_score(self, X, y):
        predictions = self.model.predict(X)
        accuracy = self.model.score(X, y)
        return predictions, accuracy

    def predict_and_score(self, data_test):
        X, y = self.get_x_y_lists(data_test)
        self.X_vecs = X
        self.y_true_vec = y
        predictions, accuracy = self._model_predict_and_score(X, y)
        f1_micro = f1_score(y, predictions, average='micro')
        f1_macro = f1_score(y, predictions, average='macro')
        f1_first_class = max(f1_score(y, predictions, average=None))
        cm = confusion_matrix(y, predictions)
        class_count = Counter(y)
        self.results = {'predictions': predictions,
                        'accuracy': accuracy,
                        'f1_micro': f1_micro,
                        'f1_macro': f1_macro,
                        'f1': f1_first_class,
                        'confusion_matrix': cm,
                        'class_count': class_count}

    def show_results(self):
        super().show_results()

        # for classification, show a bar plot showing the average scores per type in each class:
        # only when the X vector contains values for all types (the "all" type group)
        if len(self.X_vecs[0]) == len(Utils_consts.TYPES_ORDERED):
            # average the vectors of each class:
            X_per_class = Utils.get_combined_X(self.X_vecs, self.y_true_vec)
            # show plot:
            Utils.show_vector_bar_plot(X_per_class)


class MyLinearRegression(Model):
    def __init__(self):
        super()
        self.model_name = 'LinearRegression'
        self.score_type = 'gold_score'
        self.input_type = 'vector'
        
    def _model_init(self):
        return LinearRegression()
        
    def _model_fit(self, model_intialized, X, y):
        return model_intialized.fit(X, y)
    
    def _model_predict_and_score(self, X, y):
        predictions = self.model.predict(X)
        regression_score = self.model.score(X, y)
        return predictions, regression_score

    def predict_and_score(self, data_test):
        X, y = self.get_x_y_lists(data_test)
        self.X_vecs = X
        self.y_true_vec = y
        predictions, regression_score = self._model_predict_and_score(X, y)
        pearson = pearsonr(predictions, y)
        mse = mean_squared_error(y, predictions)
        ndcg1 = self.ndcg_at_k(y, predictions, 1)
        mAP = self.mean_average_precision([d['gold_class'] for d in data_test], predictions)
        self.results = {'predictions': predictions,
                        'regression_score': regression_score,
                        'pearson': pearson,
                        'mean_squared_error': mse,
                        'NDCG@1': ndcg1,
                        'mAP': mAP}

    def show_results(self):
        super().show_results()

    def ndcg_at_k(self, y_true, y_pred, ndcg_k):
        return ndcg_score([y_true], [y_pred], k=ndcg_k)

    def mean_average_precision(self, y_true, y_pred):
        # the mean average precision is the mean of AP of the different available classes:
        # y_true = [d['gold_class'] for d in data]
        classes = set(y_true)
        APs = {}
        for c in classes:
            # for the current class, set y_true to 1 if it's the class, or 0 if not:
            ys_for_c = [(int(y_t == c), y_p) for y_t, y_p in zip(y_true, y_pred)]
            y_true_c, y_pred_c = zip(*ys_for_c)
            APs[c] = average_precision_score(list(y_true_c), list(y_pred_c))
        # print(f'AP Debug: {APs}')
        return np.mean(list(APs.values()))
        
        
class MyBERTClassifier(MySVMClassifier):
    def __init__(self):
        super()
        self.model_name = 'BERTClassifier'
        self.score_type = 'gold_class'
        self.input_type = 'text'
        
        self.base_model_name = "FacebookAI/roberta-base"
        #self.learning_rate = 2e-5
        self.max_length = 256
        self.batch_size = 8
        self.epochs = 5

    def _model_init(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels=2)
        return (model, tokenizer)
        
    def _model_fit(self, model_intialized, X, y):
        model, tokenizer = model_intialized
    
        training_args = TrainingArguments(
            output_dir="models/roberta-fine-tuned-classifier",
            #learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            #metric_for_best_model="mse",
            load_best_model_at_end=True,
            #weight_decay=0.01,
        )
        
        dataset = self.prepare_data_hf(X, y, tokenizer, split_to_train_val=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            #compute_metrics=compute_metrics_for_regression_hf,
        )

        trainer.train()
        return {'trainer': trainer, 'model': model, 'tokenizer': tokenizer}
    
    def _model_predict_and_score(self, X, y):
        tokenizer = self.model['tokenizer']
        model = self.model['model']
        
        dataset = self.prepare_data_hf(X, y, tokenizer, split_to_train_val=False)['data']
        y_preds = []
        y_actual = []
        y_preds_labels = []

        for i in range(len(dataset)):
            input_text = dataset[i]["text"]
            input_label = dataset[i]["label"]
            encoded = tokenizer(input_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
            with torch.no_grad():
                y_pred_current = model(**encoded).logits.argmax().item()
                y_preds.append(y_pred_current)
                y_preds_labels.append(Utils_consts.ID_TO_LABEL[y_pred_current])
            y_actual.append(input_label)
    
        accuracy = accuracy_score(y_preds, y_actual)
        
        return y_preds_labels, accuracy
        
    def preprocess_function_hf(self, example, tokenizer):
        val = Utils_consts.LABEL_TO_ID[example["label"]]
        example = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
        example["label"] = val    
        return example
        
    def prepare_data_hf(self, X, y, tokenizer, split_to_train_val=False):
        data_list_raw = [{"text": text, "label": label} for text, label in zip(X, y)]
        data_orig = Dataset.from_list(data_list_raw)
        data_orig = data_orig.map(self.preprocess_function_hf, fn_kwargs={"tokenizer": tokenizer})
        if split_to_train_val:
            train_valid_sets = data_orig.train_test_split(test_size=0.2)
            dataset = DatasetDict({
                "train": train_valid_sets['train'],
                "val": train_valid_sets['test']
            })
        else:
            dataset = DatasetDict({"data": data_orig})
        return dataset