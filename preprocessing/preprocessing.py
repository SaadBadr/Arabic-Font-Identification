#!/usr/bin/env python
# coding: utf-8

# Hello everyone,
# 
# In this module, it is required to implement the preprocessor for Arabic Font Identification System.
# 
# At first, what are the main steps that we should go through in this module?
# 
# 
# # TODOs:
# 
# 1. Understand the problem
# 2. Binarization
# 3. Extract Edges
# 4. Extract Skeleton
# 5. Extract Diacritics
# 6. Extract Text only
# 7. Testing

# In[1]:


##################################################### imports #####################################################


# # 1. Understand the problem
# 
# We are interested only in the morphology of text letters, all images are first converted into binary (i.e., black text on a white background). It should be known that most images contain either meaningful texture, which is a part of the decoration or a meaningless one that resulted from the noise while capturing. In either case, we illuminate the background so it does not affect the results.
# 
# - Input: --
# - Output: --

# # 2. Binarization

# In[2]:


def binarize():
    pass


# # 7. Testing

# In[3]:


def testing():
    pass


# In[4]:


if __name__ == '__main__':
    testing()


# In[5]:


def create_py():
    get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb')


# In[6]:


if __name__ == '__main__':
    create_py()

