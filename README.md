# Heart-disease-prediction-using-AWS-sagemaker
Heart Disease Prediction using SageMaker

Overview

This project demonstrates a machine learning workflow to predict heart disease using the AWS SageMaker platform. The workflow includes data preprocessing, model training, deployment, and real-time inference. The Linear Learner algorithm is utilized for binary classification.

Project Structure

1. Setup and Configuration

S3 Bucket: The training data (heart.csv) is stored in an S3 bucket.

Example Bucket: data-prj-sde

Data Path: s3://data-prj-sde/sagemaker/heart/heart.csv

IAM Role: AWS IAM role with permissions to access S3 and SageMaker.

2. Code Files

The primary script includes:

Loading the dataset from S3.

Preparing the data for training.

Training the model using the Linear Learner algorithm.

Deploying the model as an endpoint.

Performing real-time predictions.

Steps to Run the Project

1. Prerequisites

AWS Account with SageMaker and S3 access.

Python installed with the following libraries:

boto3

sagemaker

pandas

numpy

2. Prepare the Dataset

Ensure your data (heart.csv) is uploaded to the specified S3 bucket.

The dataset should contain 14 columns: 13 features and 1 target (label).

Features include: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

Target: 0 (No Heart Disease) or 1 (Heart Disease).

3. Execution Steps

Step 1: Define Bucket and Data Location

Specify the S3 bucket and file location:

bucket = 'data-prj-sde'
prefix = 'sagemaker/heart'
data_key = 'heart.csv'
data_location = f's3://{bucket}/{data_key}'

Step 2: Load Data and Preprocess

Load the dataset into a Pandas DataFrame.

Separate features (columns 0-12) and target (column 13).

Convert the data into the RecordIO protobuf format required by SageMaker.

Step 3: Train the Model

Use the SageMaker Linear Learner algorithm for binary classification.

Set hyperparameters:

feature_dim=13

predictor_type='binary_classifier'

mini_batch_size=100

Example training code:

linear = sagemaker.estimator.Estimator(container,
                                       role,
                                       train_instance_count=1,
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=13,
                           predictor_type='binary_classifier',
                           mini_batch_size=100)
linear.fit({'train': s3_train_data})

Step 4: Deploy the Model

Deploy the trained model as an endpoint:

heartdisease_predictor = linear.deploy(initial_instance_count=1,
                                       instance_type='ml.m4.xlarge')

Step 5: Make Predictions

Pass feature vectors to the endpoint for inference.

Configure serializers and deserializers for input and output:

from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

heartdisease_predictor.content_type = 'text/csv'
heartdisease_predictor.serializer = CSVSerializer()
heartdisease_predictor.deserializer = JSONDeserializer()

result = heartdisease_predictor.predict(vectors[5][0:13])
print(result)

Step 6: Cleanup

Delete the endpoint after use to avoid incurring charges:

import sagemaker
sagemaker.Session().delete_endpoint(heartdisease_predictor.endpoint)

Results

The model predicts heart disease with a binary output (0 or 1).

Example prediction result:

{
    "predictions": [
        {
            "predicted_label": 1.0,
            "score": 0.89
        }
    ]
}

predicted_label: 1.0 indicates disease presence.

score: Confidence score for the prediction.

Key Notes

Always validate the input data format before training or inference.

Use smaller instance types (e.g., ml.t2.medium) during testing to save costs.

Ensure the endpoint is deleted after predictions to avoid additional charges.

Troubleshooting

S3 Access Issues: Ensure the bucket and object permissions allow SageMaker access.

Training Errors: Check SageMaker logs in the AWS Management Console for detailed error messages.

Deployment Failures: Verify that the correct instance type is specified and sufficient resources are available.

References

AWS SageMaker Documentation

Pandas Documentation

NumPy Documentation

