# vertex_mnist_pipeline.py

from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component, Input, Output, Dataset, Model
from google_cloud_pipeline_components import aiplatform as gcc_aip

PROJECT_ID = 'your-gcp-project-id'
REGION = 'us-central1'
BUCKET_NAME = 'your-bucket-name'
PIPELINE_ROOT = f'gs://{BUCKET_NAME}/pipeline_root/'

@component
def preprocess_data_op(output_data: Output[Dataset]):
    import pandas as pd
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os

    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)

    np.savez(os.path.join(output_data.path, 'mnist_preprocessed.npz'),
             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

@component(base_image='python:3.9', packages_to_install=['scikit-learn'])
def train_model_op(data: Input[Dataset], model: Output[Model]):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import joblib
    import os

    with np.load(os.path.join(data.path, 'mnist_preprocessed.npz')) as f:
        X_train, y_train = f['X_train'], f['y_train']

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    model_path = os.path.join(model.path, 'model.joblib')
    joblib.dump(clf, model_path)

@pipeline(name="mnist-vertex-pipeline", pipeline_root=PIPELINE_ROOT)
def mnist_pipeline():
    preprocessing_task = preprocess_data_op()
    training_task = train_model_op(data=preprocessing_task.outputs["output_data"])

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_pipeline.json",
    )
