# **Machine Learning**

## **Data Understanding**
  In the model development process, the dataset used is the Indonesian tourism data obtained from Kaggle, compiled by the GetLoc team, one of the Bangkit Academy 2021 Capstone Project teams. This dataset contains information about tourist attractions from several cities in Indonesia, namely Jakarta, Surabaya, Semarang, Bandung, and Yogyakarta. The dataset consists of 4 files:

1. `pariwisata_ dengan _id.csv`: Contains information about tourist attractions in 5 major cities in Indonesia, totaling ~400 entries.
2. `user.csv`: Includes dummy user data to create recommendation features based on users.
3. `Tourism_rating.csv`: Contains 3 columns—user, place, and the given rating—used to create a recommendation system based on these ratings.
4. `package_tourism.csv`: Includes recommendations for nearby places based on time, cost, and rating.

## **Data Preparation**
  In this stage, the process involves checking for missing values and descriptive statistics. Subsequently, file merging is performed to consolidate the data, aligning it with the model development objectives.

# **Modeling**

## **Content-Based Filtering Model Development**
We implemented count vectorization to convert important features of each tourist spot into vector representations. Subsequently, we calculated the similarity between items using cosine similarity, measuring the degree of similarity between tourist spots based on the count vectorization matrix previously created. As a result, we created a dataframe from the cosine similarity calculations, with rows and columns representing the names of tourist spots. Next, we created the `rekomendasi_wisata_by_keyword` function to display content-based recommendation results, including the names of tourist spots with desired categories based on the city's name.

## **Collaborative Filtering Model Development**
This technique requires user or reader rating data. Collaborative filtering is a method in recommendation systems that predicts user preferences for items based on information from other users (collaboration). Our collaborative filtering model focuses on user similarity (User-Based Collaborative Filtering).
First, we divided the data into training and validation sets and proceeded with the model training process. During training, the model calculated the matching scores between users and tourist spots using embedding techniques. We performed embedding on user and tourist spot data, then performed dot product multiplication operations between user embedding and book titles. Additionally, biases were added for each user and book title. The matching scores were set on a scale of [0,1] using the sigmoid activation function. Our model was created using the `RecommenderNet` class from the Keras Model class, utilizing Binary Crossentropy for loss function calculation, Adam as the optimizer, and Root Mean Squared Error (RMSE) as the evaluation metric.
Based on the training results, we obtained an RMSE value of around 0.2939, and RMSE on the validation data was around 0.3353. These values indicate a good quality model for the recommendation system. To obtain tourist spot recommendations, we randomly sampled users who had never visited certain tourist spots. These spots became the recommendations from the system. We used the `model.predict()` function from the Keras library to get recommendations, and the output displayed the top-10 recommendations based on user preferences and top-10 recommendations based on ratings.

# **Model Evaluation**

## **Content-Based Filtering**
We used Precision, Recall, and F1-Score as metrics to evaluate the content-based filtering model. These metrics are commonly used to measure model performance. Precision measures the ratio of relevant items produced by the model to the total items produced. Recall measures the ratio of relevant items produced by the model to the total items that should be recommended. F1 Score is a combination of Precision and Recall, providing a single value that measures the balance between the two. Before calculating the evaluation metrics using precision, recall, and F1 score, we needed ground truth data consisting of actual labels used to assess the model's prediction results. This ground truth data was created using cosine similarity results, where each row and column represented book titles, and values in each cell represented labels. The value 1 indicated similarity, and 0 indicated dissimilarity. A threshold value of 0.5 was set, and the ground truth matrix was created using the `np.where()` function from NumPy. After creating the matrix, the results were presented in the form of a dataframe, with rows and columns indexed using book titles from the data.
After creating the ground truth matrix containing actual labels from cosine similarity results, we proceeded to calculate model evaluation metrics with precision, recall, and F1 score. First, we imported the `precision_recall_fscore_support` function from the Sklearn library, which is used to calculate precision, recall, and F1 score. Due to memory allocation limitations on the device, we only took about 10,000 samples from the cosine similarity and ground truth matrices to speed up the calculation process, especially considering the relatively large matrix size. Subsequently, the cosine similarity and ground truth matrices were converted into one-dimensional arrays to facilitate comparison and metric calculation.
The results were stored in the predictions array. Finally, the `precision_recall_fscore_support` function was used to calculate precision, recall, and F1 score. The `average='binary'` parameter was used because we were measuring performance in the context of binary classification (1 or 0). The `zero_division=1` parameter was used to avoid division by zero if there were classes not present in predictions. The evaluation metric results were as follows:

- Precision: 1.0
- Recall: 1.0
- F1-score: 1.0

The evaluation results showed that the content-based filtering model provided recommendations very effectively.

## **Collaborative Filtering**
The metric used to evaluate the collaborative filtering model in this project was Root Mean Squared Error (RMSE). RMSE is commonly used to measure how well a model predicts continuous values by comparing predicted values with actual values. Based on the obtained RMSE value, the collaborative filtering model demonstrated a fairly high accuracy in predicting user preferences for items.
