# Sentiment Analysis

This project provides hands-on experience with text representations and the use of text classification for sentiment analysis. Sentiment analysis is extensively used to study customer behaviors using reviews and survey responses, online and social media, and healthcare materials for marketing and customer service applications.

## Data Preparation

We used the Amazon reviews dataset, which contains real reviews for kitchen products sold on Amazon. The dataset is downloadable [here](https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz).

### Steps:
1. **Read the Data**: Loaded the dataset as a Pandas DataFrame and retained only the Reviews and Ratings fields.
2. **Binary Label Creation**: Generated binary labels by assuming ratings above 3 indicate positive sentiment (mapped to 1) and ratings 2 or below indicate negative sentiment (mapped to 0). Discarded reviews with a rating of 3 as neutral.
3. **Dataset Statistics**: Reported the statistics of the ratings, i.e., the count of reviews for each rating.
4. **Dataset Downsizing**: Selected 100,000 reviews with positive sentiment and 100,000 reviews with negative sentiment to avoid computational burden.
5. **Dataset Splitting**: Split the dataset into 80% training and 20% testing sets.

## Data Cleaning

Performed the following cleaning steps:
- Converted all reviews to lowercase.
- Removed HTML tags and URLs.
- Removed non-alphabetical characters.
- Removed extra spaces.
- Expanded contractions (e.g., won’t → will not).

## Preprocessing

Using the NLTK package, we:
- Removed stop words.
- Performed lemmatization.

## Feature Extraction

Used sklearn to extract TF-IDF features, resulting in a dataset with features and binary labels for the selected reviews.

## Models and Evaluation

Trained and evaluated several models using the sklearn built-in implementations:

### Perceptron
- **Training and Testing Metrics**: Reported Accuracy, Precision, Recall, and F1-score.

### SVM (Support Vector Machine)
- **Training and Testing Metrics**: Reported Accuracy, Precision, Recall, and F1-score.

### Logistic Regression
- **Training and Testing Metrics**: Reported Accuracy, Precision, Recall, and F1-score.

### Multinomial Naive Bayes
- **Training and Testing Metrics**: Reported Accuracy, Precision, Recall, and F1-score.

## Outputs

When you run the `.py` file, the following outputs are printed, each on a separate line:
- Statistics of three classes (with commas between values).
- Average length of reviews before and after data cleaning (with commas between values).
- Average length of reviews before and after preprocessing (with commas between values).
- Accuracy, Precision, Recall, and F1-score for training and testing splits for each model (with commas between values).

In the Jupyter notebook, the Accuracy, Precision, Recall, and F1-score for the models are printed in separate lines.

---

This project showcases practical applications of sentiment analysis by training multiple models and evaluating their performance on a real-world dataset.
