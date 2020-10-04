#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn


# In[3]:


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


# In[4]:


digits = datasets.load_digits()


# In[5]:


dir(digits)


# In[11]:


digits


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[12]:


X,y=digits['images'],digits['target']
X.shape


# In[13]:


y.shape


# In[14]:


some_img=X[360]
n_samples = len(some_img)
some_digits_img=some_img.reshape(n_samples,-1)
plt.imshow(some_digits_img, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()


# In[15]:


y[360]


# In[16]:


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.1,shuffle=False)


# In[26]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
predicted=lr.predict(X_test)


# In[28]:


_, axes = plt.subplots(2, 2)
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
disp = metrics.plot_confusion_matrix(lr, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()


# In[40]:


lr.predict([digits.data[340]])


# In[41]:


digits.target[340]


# In[53]:


y_pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[58]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('y_pred')
plt.ylabel('y_truth')


# In[65]:


lr.score()


# In[60]:


import joblib


# In[62]:


filename = 'HandWrittenDigitsClassification'
joblib.dump(lr, filename)


# In[64]:


loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)


# In[ ]:




