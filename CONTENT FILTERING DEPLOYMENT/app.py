import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Copy semua kode pre-processing di sini
# ...
rating = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_rating.csv')
tourism = pd.read_csv('https://raw.githubusercontent.com/AqillaSM/DatasetNavomobility/main/tourism_with_id.csv')

all_tourism = pd.merge(rating,tourism[["Place_Id","Place_Name","Description","City","Category", "Rating"]],on='Place_Id', how='left')

all_tourism['city_category'] = all_tourism[['City','Category']].agg(' '.join,axis=1)

datapreparation= all_tourism.drop_duplicates("Place_Id")

place_id = datapreparation.Place_Id.tolist()

place_name = datapreparation.Place_Name.tolist()

place_category = datapreparation.Category.tolist()

place_desc = datapreparation.Description.tolist()

place_rat = datapreparation.Rating.tolist()

place_city = datapreparation.City.tolist()

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

"""# Content Based Filtering"""

data_content = tourism_complete

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

cv.fit(data_content['city_category'])

print("Features Name: ", list(cv.vocabulary_.keys()))

cv_matrix = cv.transform(data_content['city_category'])

pd.DataFrame(
    cv_matrix.todense(),
    columns=list(cv.vocabulary_.keys()),
    index = data_content.name
).sample(5)

#Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity_data = cosine_similarity(cv_matrix)

similarity_df = pd.DataFrame(similarity_data,index=data_content['name'],columns=data_content['name'])


def tourism_recommendations(place_name, similarity_data=similarity_df, items=data_content[['name', 'category', 'description', 'city', 'rating']], k=15):
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k + 2):-1]]

    closest = closest.drop(place_name, errors='ignore')

    # Data manual yang akan ditambahkan
    new_data = tourism_complete[tourism_complete['name'] == place_name].head(1)

    items = pd.DataFrame(closest).merge(items).head(k)

    return pd.concat([new_data, items], ignore_index=True)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        place_name = request.form['place_name'].title()  # Menggunakan title() di sini
        if place_name not in data_content['name'].values:
            return render_template('recommendation.html', alert_message="Tidak ada wisata")
        else:
            recommendations = tourism_recommendations(place_name)
            return render_template('recommendation.html', recommendations=recommendations.to_dict(orient='records'))
    return render_template('recommendation.html', recommendations=[])


if __name__ == '__main__':
    app.run(debug=True)


