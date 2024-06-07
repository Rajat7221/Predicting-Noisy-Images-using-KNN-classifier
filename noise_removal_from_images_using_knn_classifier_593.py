#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Predicting Noisy Images


# In[1]:


# Step 1)

import numpy as np
import pandas as pd
import gzip 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
import matplotlib.pyplot as plt


# In[2]:


# Step 2) Define the Function to Show Images

def showImage(data):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28)
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()



# In[3]:


# Step 3) Load the Data

dtapath ="/cxldata/datasets/project/mnist/"

#Provide the path to the files:

filePath_train_set = '/cxldata/datasets/project/mnist/train-images-idx3-ubyte.gz'
filePath_train_label = '/cxldata/datasets/project/mnist/train-labels-idx1-ubyte.gz'
filePath_test_set = '/cxldata/datasets/project/mnist/t10k-images-idx3-ubyte.gz'
filePath_test_label = '/cxldata/datasets/project/mnist/t10k-labels-idx1-ubyte.gz'


#Open the Gzip files:

with gzip.open(filePath_train_label, 'rb') as trainLbpath:
     trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                               offset=8)
with gzip.open(filePath_train_set, 'rb') as trainSetpath:
     trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
     testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8,
                               offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
     testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(testLabel), 784)


        
# Store the data into the 4 variables:

X_train, X_test , y_train, y_test = trainSet, testSet, trainLabel, testLabel        


# In[4]:


# Step 4)  Explore the Data

# First, I will print the shape of the variables I created in the previous step:

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# In[5]:


# Next, I will use the showImage() function I created earlier to view the first image in the training set and it's corresponsing label:

showImage(X_train[0])
y_train[0]


# In[6]:


# Finally, we will view few more images from the dataset:

plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    array_image = X_train[i].reshape(28, 28)
    plt.imshow(array_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.show()


# In[7]:


# Step 5) Shuffle the training Data

np.random.seed(42)

#Now we shuffle the training data:

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[8]:


# Step 6) Add Noise to the Data

'''
In this step, we will add noise to the images. We would use the randint function
to generate the noise and then add the noise to the train and test set. 
Finally, we would store this noisy data in 2 new variables 
called X_train_mod and X_test_mod.
'''

import numpy.random as rnd

# Generate the noise and store the noisy data into the new variables:

noise_train = rnd.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test






# In[9]:


'''
If you notice the code, the first line generates the noise by using the randint() function. 
The randint() function takes 4 inputs:

low
high
size
dtype
Here, low and high gives the range of the distribution, whereas size defines the output shape, 
and finally, dtype specifies the data type. It gives an array of integers as an output. 
Next we add this noise to the X-train and X_test datasets. 
Finally, we create a new set of labels, these labels are the original images. 
The model we will create will predict these images from their corresponding noisy image.

'''


# In[10]:


# View the Noisy Data

#Use the showImage() function we created earlier to view the noisy images and it's corresponding image without the noise:



showImage(X_test_mod[4000])

showImage(y_test_mod[4000])


# In[11]:


# Step 7) Train a KNN Classifier on Noisy images so tht it can predict non-noisy image from the same 

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)


# In[12]:


# Step 8) Predict Noisy Image


clean_digit = knn_clf.predict([X_test_mod[5000]])
showImage(clean_digit)

# original img
showImage(y_test_mod[5000])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




