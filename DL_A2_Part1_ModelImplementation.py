#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
import math
from dl_toolkit import MLPClassifier


# In[2]:


train_set=pd.read_pickle('train_set.pkl')
val_set=pd.read_pickle('val_set.pkl')
training_features=np.array(np.array(train_set['Image'][0]).flatten())
for i in range(1,train_set.shape[0]):
    training_features=np.vstack([training_features,np.array(train_set['Image'][i]).flatten()])
training_targets=np.array(train_set['Labels'])
validation_features=np.array(np.array(val_set['Image'][0]).flatten())
for i in range(1,val_set.shape[0]):
    validation_features=np.vstack([validation_features,np.array(val_set['Image'][i]).flatten()])
validation_targets=np.array(val_set['Labels'])


# In[4]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='sigmoid',optimizer='gradient_descent',Weight_init='random',Batch_size=training_features.shape[0],Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('sigmoid-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'sigmoid-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'sigmoid-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'sigmoid-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'sigmoid-roc-curve-validation.png')
pickle.dump(MLPC,open('model-sigmoid.pkl','wb'))
model=pickle.load(open('model-sigmoid.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[3]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='tanh',optimizer='gradient_descent',Weight_init='random',Batch_size=training_features.shape[0],Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('tanh-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'tanh-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'tanh-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'tanh-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'tanh-roc-curve-validation.png')
pickle.dump(MLPC,open('model-tanh.pkl','wb'))
model=pickle.load(open('model-tanh.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[7]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='relu',optimizer='gradient_descent',Weight_init='random',Batch_size=training_features.shape[0],Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('relu-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'relu-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'relu-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'relu-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'relu-roc-curve-validation.png')
pickle.dump(MLPC,open('model-relu.pkl','wb'))
model=pickle.load(open('model-relu.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[6]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='tanh',optimizer='momentum',Weight_init='random',Batch_size=64,Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('momentum-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'momentum-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'momentum-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'momentum-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'momentum-roc-curve-validation.png')
pickle.dump(MLPC,open('model-momentum.pkl','wb'))
model=pickle.load(open('model-momentum.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[7]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='tanh',optimizer='nag',Weight_init='random',Batch_size=64,Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('nag-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'nag-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'nag-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'nag-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'nag-roc-curve-validation.png')
pickle.dump(MLPC,open('model-nag.pkl','wb'))
model=pickle.load(open('model-nag.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[4]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='tanh',optimizer='adagrad',Weight_init='random',Batch_size=64,Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('adagrad-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'adagrad-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'adagrad-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'adagrad-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'adagrad-roc-curve-validation.png')
pickle.dump(MLPC,open('model-adagrad.pkl','wb'))
model=pickle.load(open('model-adagrad.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[5]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.1,activation_function='tanh',optimizer='adam',Weight_init='random',Batch_size=64,Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('adam-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'adam-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'adam-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'adam-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'adam-roc-curve-validation.png')
pickle.dump(MLPC,open('model-adam.pkl','wb'))
model=pickle.load(open('model-adam.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[6]:


MLPC=MLPClassifier(layers=[784,200,80,10],learning_rate=0.001,activation_function='tanh',optimizer='rmsprop',Weight_init='random',Batch_size=64,Num_epochs=200,dropout=0)
MLPC.fit(training_features,training_targets,validation_features,validation_targets)
MLPC.loss_plot('rmsprop-loss-plot.png')
MLPC.confusion_matrix_plot(training_targets,MLPC.predict(training_features),'rmsprop-confusion-matrix-training.png')
MLPC.confusion_matrix_plot(validation_targets,MLPC.predict(validation_features),'rmsprop-confusion-matrix-validation.png')
MLPC.roc_plot(training_features,training_targets,'rmsprop-roc-curve-training.png')
MLPC.roc_plot(validation_features,validation_targets,'rmsprop-roc-curve-validation.png')
pickle.dump(MLPC,open('model-rmsprop.pkl','wb'))
model=pickle.load(open('model-rmsprop.pkl','rb'))
print('Validation Accuracy =',model.score(validation_features,validation_targets))


# In[ ]:


# model=pickle.load(open('model-rmsprop.pkl','rb'))

