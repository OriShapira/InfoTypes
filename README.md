# Information Types in Product Reviews
This repository contains the code and resources used in the paper "Information Types in Product Reviews". This includes the typology prediction model, evaluation of results and analyses. 

## Analyses and Experiments

### DownstreamPredictions
Each script separately predicts a task based on the types in the text:
* PredictionReviewHelpfulness.py: How well do the types in a review predict its helpfulness
* PredictionReviewSentiment.py: How well do the types in a review predict its sentiment
* PredictionSentenceHelpfulness.py: How well do the types in a sentence predict its helpfulness
The bottom of each file has a main that trains and tests the respective downstream task.
To run, e.g.,:
```
>>> cd DownstreamPrediction
>>> python PredictionReviewHelpfulness.py
```

### Evaluation
Each script evaluates the type predictions:
* EvaluationFull.py: evaluation on the test set
* EvaluationOnTypeTip.py: evaluation on a Tips benchmark
* EvaluationOnTypeOpinions.py: evaluation on an opinions benchmark

### Analysis
Each script separately computes the analyses for the three analyses:
* AnalysisReviewVsSummary.py: Types in Reviews vs. in Summaries
* AnalysisProductCategory.py: Types in different product categories
* AnalysisRhetoricalStructure.py: Types viewed throughout a review or summary, as a rhetorical structure

### How to Run
To run any of the above scripts, run it from the library of that script, e.g.,
```
>>> cd Evaluation
>>> python EvaluationFull.py
```

You may need to add this to the top of the script if run from command line:
```
import sys
sys.path.append('..')
```

Notice, all scripts run without arguments. To change any of the parameters, do it inside the script, usually at the
bottom of the file in the main section.

The three downstream tasks also train/test a RoBERTa model, which can run on a CPU, but is very slow. It is recommended
to run these on a machine with a GPU to significantly speed up the process.

## TypePrediction
The code for running an LLM to predict the types of product-related sentences.
The script is best run on a machine with GPUs. If a larger model is run (which is likely the case, e.g., flan-t5-xxl), a strong machine is required (e.g., with 4 GPUs and at least 32GB GPU Memory).
The script is standalone and is not related to any of the other code in the repository.

First install the requirements in predict_types_with_llms_requirements.txt, then run:
```
python predict_types_with_llms.py
```
You can change any of the parameters near the top of the file.

## Citation
If you use any of the code here, please cite: 
```
TODO
```

## License

>   Licensed under the Apache License, Version 2.0 (the "License").
>   You may not use this file except in compliance with the License.
>   You may obtain a copy of the License at
>   
>       http://www.apache.org/licenses/LICENSE-2.0
>   
>   Unless required by applicable law or agreed to in writing, software
>   distributed under the License is distributed on an "AS IS" BASIS,
>   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
>   See the License for the specific language governing permissions and
>   limitations under the License.