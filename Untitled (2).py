#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Enter bucket name
bucket = 'data-prj-sde'
prefix = 'sagemaker/heart'

#Enter data file name (e.g. heart.csv)
data_key = 'heart.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# In[4]:


import pandas as pd
import json

# read the data from S3
heart_data = pd.read_csv(data_location)

#print out a sample of data.
heart_data.head()


# In[5]:


import numpy as np
vectors = np.array(heart_data).astype('float32')

#target column - value must be either 0 or 1
labels = vectors[:,13]
print ("label data is")
print (labels)


#drop the target column.  Use the features as part of the training data
training_data = vectors[:, :13]
print ("Training data is")
print (training_data)


# In[6]:


import io
import os
import sagemaker.amazon.common as smac

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, training_data, labels)
buf.seek(0)

key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[7]:


output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# In[8]:


from sagemaker.amazon.amazon_estimator import get_image_uri
import sagemaker

container = get_image_uri(boto3.Session().region_name, 'linear-learner', "latest")

sess = sagemaker.Session()
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


# In[9]:


heartdisease_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')


# In[12]:


from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

heartdisease_predictor.content_type = 'text/csv'
heartdisease_predictor.serializer = CSVSerializer()
heartdisease_predictor.deserializer = JSONDeserializer()


# In[13]:


print('Endpoint name: {}'.format(heartdisease_predictor.endpoint))


# In[14]:


vectors[5][0:13]


# In[15]:


result = heartdisease_predictor.predict(vectors[5][0:13])
print(result)


# In[ ]:


import sagemaker

sagemaker.Session().delete_endpoint(heartdisease_predictor.endpoint)

