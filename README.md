# recsys20challenge-nlp

Prequisites:
- Spark
- Polynote
- Scala
- Python
- A lot of CPU, RAM and Storage ;-)

Note that you need to configure spark using the polynote configurations.

## 0. Sample Training Data (optional)

Sample training data using `dataset_sampling.ipynb`.

## 1. Preprocess Data

Transform all tsv files with `preprocessing_polynote_scala.ipynb`.  
Afterwards load the embeddings with `python/run_use.py`.  

## 2. Fit Model and Predict

Fit the training data using `train_pipeline_stage2.ipynb`.  
Run the prediction on the validation/test data using `run_pipeline_stage2.ipynb`.  
