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

Transform all tsv files with `preprocessing.ipynb`.  

## 2. Compute Embeddings

1. Create a virtualenv using Python 3.8.
2. Navigate to the `python` subdirectory.
3. Install all dependencies: `pip install -r requirements.txt`.
4. Edit `compute_embeddings.py` to configure the input dataset.
5. Run `compute_embeddings.py` within your virtualenv.

## 2. Train and Run Classifier

Fit the training data using `train_classifier.ipynb`.  
Run the prediction on the validation/test data using `run_classifier.ipynb`.  
