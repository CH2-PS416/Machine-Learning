
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from flask import Flask, jsonify, render_template

from tensorflow import keras
from tensorflow.keras import layers
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)
#load dataset
rating = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_rating.csv')
tourism = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_with_id.csv')

all_tourism = pd.merge(rating,tourism[["Place_Id","Place_Name","Description","City","Category", "Rating"]],on='Place_Id', how='left')

all_tourism['city_category'] = all_tourism[['City','Category']].agg(' '.join,axis=1)

datapreparation= all_tourism.drop_duplicates("Place_Id")

place_id = datapreparation.Place_Id.tolist()

place_name = datapreparation.Place_Name.tolist()

place_category = datapreparation.Category.tolist()

place_desc = datapreparation.Description.tolist()

place_city = datapreparation.City.tolist()

place_rat = datapreparation.Rating.tolist()

city_category = datapreparation.city_category.tolist()

tourism_complete = pd.DataFrame({
    "id":place_id,
    "name":place_name,
    "category":place_category,
    "description":place_desc,
    "city": place_city,
    "rating": place_rat,
    "city_category":city_category
})

datarating = rating

user_ids = datarating.User_Id.unique().tolist()

user_to_user_encoded = {x:i for i, x in enumerate(user_ids)}

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

place_ids = datarating.Place_Id.unique().tolist()

place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

place_encoded_to_place = {x: i for x, i in enumerate(place_ids)}

datarating['user'] = datarating.User_Id.map(user_to_user_encoded)

datarating['place'] = datarating.Place_Id.map(place_to_place_encoded)

num_users = len(user_to_user_encoded)

num_place = len(place_encoded_to_place)

datarating['Place_Ratings'] = datarating['Place_Ratings'].values.astype(np.float32)

min_rating = min(datarating['Place_Ratings'])

max_rating= max(datarating['Place_Ratings'])

print('Number of User: {}, Number of Place: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_place, min_rating, max_rating
))

datarating = datarating.sample(frac=1,random_state=42)

x = datarating[['user','place']].values

y = datarating['Place_Ratings'].apply(lambda x:(x-min_rating)/(max_rating-min_rating)).values

train_indices = int(0.9 * datarating.shape[0])

x_train,x_val,y_train,y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

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

model = RecommenderNet(num_users, num_place, 50)

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 16,
    epochs = 50,
    validation_data = (x_val, y_val),
)

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

# Fungsi untuk memberikan rekomendasi
@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    # Logika untuk mendapatkan rekomendasi berdasarkan user_id
    # Gunakan model untuk mendapatkan rekomendasi

    # Ambil data pengguna
    user_id = int(user_id)
    place_visited_by_user = datarating[datarating['User_Id'] == user_id]

    # Mendapatkan tempat yang belum dikunjungi oleh pengguna
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user['Place_Id'].values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))

    # Prediksi menggunakan model
    ratings = model.predict(user_place_array).flatten()

    # Mendapatkan top 10 rekomendasi
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]

    # Mengambil detail tempat yang direkomendasikan
    recommend_place = place_df[place_df['id'].isin(recommended_place_ids)]

    # Membuat DataFrame untuk output rekomendasi
    recommended_place_data = []
    for row in recommend_place.itertuples():
        recommended_place_data.append([row.name, row.city, row.description, row.rating])

    output = {
        "user_id": user_id,
        "recommended_places": pd.DataFrame(recommended_place_data, columns=['Place', 'City', 'Description', 'Rating']).to_dict(orient='records')
    }
    return jsonify(output)

@app.route('/')
def index():
    # Jika ingin menambahkan logika sebelum menampilkan HTML, letakkan di sini
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

