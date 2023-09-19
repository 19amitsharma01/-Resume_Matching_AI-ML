#!/usr/bin/env python
# coding: utf-8

# In[1]:



import PyPDF2
import pdfminer
import re
import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Resume.csv")


# In[3]:


df.head()


# In[4]:



import spacy
spacy.cli.download("en_core_web_sm")


# In[5]:


import pandas as pd
import spacy
import nltk
import re

nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")


# In[6]:


def extract_category(text):
    
    sentences = nltk.sent_tokenize(text)
    if sentences:
        return sentences[0]
    else:
        return None
skills_df = pd.read_csv(r"C:\Users\hp\Downloads\cleaned_data.csv")
skills_df = skills_df.dropna(subset=['Cleaned_Value'])
skills_df['Cleaned_Value'] = skills_df['Cleaned_Value'].str.strip()
skills = skills_df['Cleaned_Value'].tolist()

def extract_skills(text):
    nlp_text = nlp(text)
    skillset = []

    
    def is_skill(token):
        if token is not None and token.text is not None:
            return any(
                skill is not None and token.text.lower() == skill.lower()
                for skill in skills
            )
        return False

    
    for token in nlp_text:
        if is_skill(token):
            skillset.append(token.text)

  
    for chunk in nlp_text.noun_chunks:
        if is_skill(chunk.root):
            skillset.append(chunk.text)

    return list(set(skillset))  

def extract_education(text):
    pattern = r"([A-Za-z]+(?: [A-Za-z]+)*) (?:in|from) ([A-Za-z]+(?: [A-Za-z]+)*)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    education = [{"degree": match[0], "institution": match[1]} for match in matches]
    return education


# In[10]:


random_subset = df.head(n=15)  

random_subset["Skills"] = random_subset["Resume_str"].apply(extract_skills)


# In[11]:


job = pd.read_csv(r"C:\Users\hp\Downloads\training_data.csv")


# In[12]:


job.head()


# In[13]:


random_jobset = job.head(15)


# In[14]:


random_jobset["Skills"] = random_jobset["job_description"].apply(extract_skills)


# In[16]:


import gensim
from gensim.models import Word2Vec


# In[21]:


model = Word2Vec(random_jobset['Skills'], vector_size=100, window=5, min_count=1, sg=0)

def get_average_vector(words, model, vector_size):
    vec_sum = sum(model.wv[word] for word in words if word in model.wv)
    return vec_sum / len(words) if len(words) > 0 else [0] * vector_size

random_jobset['skill_embeddings'] = random_jobset['Skills'].apply(lambda x: get_average_vector(x, model, vector_size=100))


# In[23]:


model = Word2Vec(random_subset['Skills'], vector_size=100, window=5, min_count=1, sg=0)

def get_average_vector(words, model, vector_size):
    vec_sum = sum(model.wv[word] for word in words if word in model.wv)
    return vec_sum / len(words) if len(words) > 0 else [0] * vector_size

random_subset['skill_embeddings'] = random_subset['Skills'].apply(lambda x: get_average_vector(x, model, vector_size=100))


# In[30]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_matrix = cosine_similarity(
    list(random_jobset['skill_embeddings']),
    list(random_subset['skill_embeddings'])
)

cosine_sim_df = pd.DataFrame(
    data=cosine_sim_matrix,
    index=random_jobset.index, 
    columns=random_subset.index  
)


# In[31]:


cosine_sim_sorted_df

