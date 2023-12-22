#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt


# In[2]:


#load dataset
rating = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_rating.csv')
tourism = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_with_id.csv')


# In[3]:


tourism


# In[4]:


rating


# # Data Preprocessing

# In[5]:


tourism.info()


# In[6]:


rating.info()


# In[7]:


rating.describe()


# In[8]:


all_tourism = pd.merge(rating,tourism[["Place_Id","Place_Name","Description","Rating","City","Category",]],on='Place_Id', how='left')
all_tourism


# In[9]:


all_tourism.isnull().sum()


# In[10]:


all_tourism['city_category'] = all_tourism[['City','Category']].agg(' '.join,axis=1)
all_tourism


# In[11]:


datapreparation= all_tourism.drop_duplicates("Place_Id")
datapreparation


# In[12]:


place_id = datapreparation.Place_Id.tolist()

place_name = datapreparation.Place_Name.tolist()

place_category = datapreparation.Category.tolist()

place_desc = datapreparation.Description.tolist()

place_rat = datapreparation.Rating.tolist()

place_city = datapreparation.City.tolist()

city_category = datapreparation.city_category.tolist()


# In[13]:


tourism_complete = pd.DataFrame({
    "id":place_id,
    "name":place_name,
    "category":place_category,
    "description":place_desc,
    "rating":place_rat,
    "city": place_city,
    "city_category":city_category
})

tourism_complete


# # Content Based Filtering

# In[14]:


data_content = tourism_complete
data_content.sample(5)


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

cv.fit(data_content['city_category'])

print("Features Name: ", list(cv.vocabulary_.keys()))


# In[16]:


cv_matrix = cv.transform(data_content['city_category'])

cv_matrix.shape


# In[17]:


cv_matrix.todense()


# In[18]:


pd.DataFrame(
    cv_matrix.todense(),
    columns=list(cv.vocabulary_.keys()),
    index = data_content.name
).sample(5)


# In[19]:


#Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity_data = cosine_similarity(cv_matrix)
similarity_data


# In[20]:


similarity_df = pd.DataFrame(similarity_data,index=data_content['name'],columns=data_content['name'])
similarity_df.sample(5,axis=1).sample(10,axis=0)


# In[21]:


def tourism_recommendations(place_name, similarity_data=similarity_df, items=data_content[['name', 'category', 'description', 'city', 'rating']], k=15):
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k + 2):-1]]

    closest = closest.drop(place_name, errors='ignore')

    # Data manual yang akan ditambahkan
    new_data = tourism_complete[tourism_complete['name'] == place_name].head(1)

    items = pd.DataFrame(closest).merge(items).head(k)

    return pd.concat([new_data, items], ignore_index=True)


# In[43]:


place_name = 'Tugu Muda Semarang'  # Menggunakan title() di sini

recommendations = tourism_recommendations(place_name)
recommendations = recommendations.drop(columns=['city_category'])


# In[44]:


recommendations


# # COLLABORATIVE FILTERING

# In[24]:


datarating = rating
datarating


# In[25]:


user_ids = datarating.User_Id.unique().tolist()

user_to_user_encoded = {x:i for i, x in enumerate(user_ids)}

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}


# In[26]:


place_ids = datarating.Place_Id.unique().tolist()

place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

place_encoded_to_place = {x: i for x, i in enumerate(place_ids)}


# In[27]:


datarating['user'] = datarating.User_Id.map(user_to_user_encoded)

datarating['place'] = datarating.Place_Id.map(place_to_place_encoded)


# In[28]:


num_users = len(user_to_user_encoded)

num_place = len(place_encoded_to_place)

datarating['Place_Ratings'] = datarating['Place_Ratings'].values.astype(np.float32)

min_rating = min(datarating['Place_Ratings'])

max_rating= max(datarating['Place_Ratings'])

print('Number of User: {}, Number of Place: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_place, min_rating, max_rating
))


# In[29]:


datarating = datarating.sample(frac=1,random_state=42)
datarating


# In[30]:


x = datarating[['user','place']].values

y = datarating['Place_Ratings'].apply(lambda x:(x-min_rating)/(max_rating-min_rating)).values

train_indices = int(0.9 * datarating.shape[0])

x_train,x_val,y_train,y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x,y)


# In[31]:


class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_place, embedding_size, dropout_rate=0, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_place = num_place
    self.embedding_size = embedding_size
    self.dropout_rate = dropout_rate
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.place_embedding = layers.Embedding(
        num_place,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.place_bias = layers.Embedding(num_place, 1)
    self.dropout = layers.Dropout(rate=dropout_rate)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_vector = self.dropout(user_vector)
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    place_vector = self.place_embedding(inputs[:, 1]) # memanggil layer embedding 3
    place_vector = self.dropout(place_vector)
    place_bias = self.place_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_place = tf.tensordot(user_vector, place_vector, 2)

    x = dot_user_place + user_bias + place_bias

    return tf.nn.sigmoid(x) # activation sigmoid


# In[32]:


model = RecommenderNet(num_users, num_place, 50)

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


# In[33]:


history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 16,
    epochs = 50,
    validation_data = (x_val, y_val),
)


# In[34]:


place_df = tourism_complete

#sample user
user_id = datarating['User_Id'].sample(1).iloc[0]
place_visited_by_user = datarating[datarating['User_Id'] == user_id]

place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user['Place_Id'].values)]['id']
place_not_visited = list(
    set(place_not_visited)
    .intersection(set(place_to_place_encoded.keys()))
)

place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(
    ([[user_encoder]] * len(place_not_visited), place_not_visited)
)


# In[35]:


ratings = model.predict(user_place_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_place_ids = [
    place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
]
top_place_user = (
    place_visited_by_user.sort_values(
        by='Place_Ratings',
        ascending=False
    )
    .head(10)['Place_Id'].values
)
place_df_rows = place_df[place_df['id'].isin(top_place_user)]

# Menampilkan rekomendasi buku dalam bentuk DataFrame
place_df_rows_data = []
for row in place_df_rows.itertuples():
    place_df_rows_data.append([row.name, row.city, row.description, row.rating])

recommend_place = place_df[place_df['id'].isin(recommended_place_ids)]

recommended_place_data = []
for row in recommend_place.itertuples():
    recommended_place_data.append([row.name, row.city, row.description, row.rating])

# Membuat DataFrame untuk output
output_columns = ['Place', 'City', 'Description', 'Rating']
df_place_visited_by_user = pd.DataFrame(place_df_rows_data, columns=output_columns)
df_recommended_place = pd.DataFrame(recommended_place_data, columns=output_columns)

# Menampilkan hasil rekomendasi dalam bentuk DataFrame
print("Showing recommendation for users: {}".format(user_id))
print("===" * 9)
print("Place with high ratings from user")
print("----" * 8)
print(df_place_visited_by_user)
print("----" * 8)
print("Top 10 Place recommendation")
print("----" * 8)
df_recommended_place


# In[36]:


recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
recommended_place


# #EVALUASI MODEL

# ##Content Based Filtering
# 

# In[37]:


# Menentukan threshold untuk mengkategorikan similarity sebagai 1 atau 0
threshold = 0.5

# Membuat ground truth data dengan asumsi threshold
ground_truth = np.where(similarity_data >= threshold, 1, 0)

# Menampilkan beberapa nilai pada ground truth matrix
ground_truth_df = pd.DataFrame(ground_truth, index=data_content['name'], columns=data_content['name']).sample(5, axis=1).sample(10, axis=0)


# In[38]:


ground_truth_df


# In[39]:


from sklearn.metrics import precision_recall_fscore_support

# Mengambil sebagian kecil dari cosine similarity matrix dan ground truth matrix
sample_size = 10000
cosine_sim_sample = similarity_data[:sample_size, :sample_size]
ground_truth_sample = ground_truth[:sample_size, :sample_size]

# Mengonversi cosine similarity matrix menjadi array satu dimensi untuk perbandingan
cosine_sim_flat = cosine_sim_sample.flatten()

# Mengonversi ground truth matrix menjadi array satu dimensi
ground_truth_flat = ground_truth_sample.flatten()

# Menghitung metrik evaluasi
predictions = (cosine_sim_flat >= threshold).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth_flat, predictions, average='binary', zero_division=1
)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# #Collaborative Filtering

# In[40]:


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

