# recsys20challenge-nlp

Prequisites:
- Spark
- Polynote
- Scala
- Python
- A lot of CPU, RAM and Storage ;-)

Note that you need to configure spark using the polynote configurations.

## Running the challenge approach

### 0. Sample Training Data (optional)

Sample training data using `challenge/dataset_sampling.ipynb`.

### 1. Preprocess Data

Transform all tsv files with `preprocessing_polynote_scala.ipynb`.  
Afterwards load the embeddings with `challenge/compute_embeddings.py`.  

### 2. Fit Model and Predict

Fit the training data using `challenge/train_pipeline_stage2.ipynb`.  
Run the prediction on the validation/test data using `challenge/run_pipeline_stage2.ipynb`.  

## Post Challenge approaches

The post challenge approaches are found in the `postchallenge` directory and require the same preprocessed parquet file as input.

## Folder structure

This repo contains 3 subdirectories:
- challenge for experiments conducted for the RecSys20 challenge using mainly Spark and Scala
- postchallenge for additional experiments after the challenge using mainly Dask and Python
- experimental several experimental notebooks, which are not relevant for the results

Additionally, there are several preprocessing notebooks that are relevant for both the challenge and postchallenge experiments.
