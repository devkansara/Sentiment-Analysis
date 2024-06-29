#Python version 3.9.10
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
nltk.download('wordnet')
nltk.download('stopwords')

#Read Data
df1=pd.read_csv('data.tsv', sep='\t', on_bad_lines='skip')

#Keep Reviews and Ratings
df2 = df1[['star_rating', 'review_headline', 'review_body']]

print(f"\n\n\nPositive Reviews: {len(df2.loc[df2['star_rating'].isin([4,5])])}, Negative Reviews: {len(df2.loc[df2['star_rating'].isin([1,2])])}, Neutral Reviews: {len(df2.loc[df2['star_rating'].isin([3])])}\n")

#We form three classes and select 100000 reviews randomly from each class.
positive_reviews=df2.loc[df2['star_rating'].isin([1,2])].sample(n=100000, random_state=36)
negative_reviews=df2.loc[df2['star_rating'].isin([4,5])].sample(n=100000, random_state=36)
processed_df = pd.concat([positive_reviews, negative_reviews])

# Print the random reviews
random_reviews = processed_df.sample(n=3, random_state=36)
print("\n")
for index, row in random_reviews.iterrows():
    print(f"Star Rating: {row['star_rating']}")
    print(f"Review Headline: {row['review_headline']}")
    print(f"Review Body: {row['review_body']}")
    print("\n" + "-"*50 + "\n") 


processed_df['label'] = processed_df['star_rating'].apply(
    lambda x: 1 if x > 3 else 0 if x < 3 else -1
    )
positive_count = len(processed_df[processed_df['label'] == 1])
negative_count = len(processed_df[processed_df['label'] == 0])
neutral_count = len(processed_df[processed_df['label'] == -1])

#Statistics of three classes (with comma between them)

processed_df['review_headline'] = processed_df['review_headline'].apply(str)
processed_df['review_body'] = processed_df['review_body'].apply(str)
processed_df['review'] = processed_df[['review_headline', 'review_body']].agg(' '.join, axis=1)
processed_df = processed_df.drop('review_headline', axis=1)
processed_df = processed_df.drop('review_body', axis=1)

print(f"\nAverage Character count before Data Cleaning: {(processed_df['review'].str.len()).mean()}",)

# Data Cleaning
# Pre-processing

# Convert to lower case
processed_df['review'] = processed_df['review'].str.lower()
# Remove HTML tags
processed_df['review'] = processed_df['review'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
# Remove URLs
processed_df['review'] = processed_df['review'].apply(lambda x: re.sub(r'http\S+', '', x))
# Remove non-alphabetical characters
processed_df['review'] = processed_df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
# Remove extra spaces
processed_df['review'] = processed_df['review'].apply(lambda x: ' '.join(x.split()))

# Performing contractions
contractions = {
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "i will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "i'd": "i would",
    "won't": "will not",
    "can't": "cannot",
    "I'm": "I am",
    "you're": "you are", 
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "we're": "we are",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "they're": "they are",
    "haven't": "have not",
    "hasn't": "has not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
}

processed_df['review'] = processed_df['review'].replace(contractions, regex=True)

print(f"\nAverage Character count after Data Cleaning: {(processed_df['review'].str.len()).mean()}\n")

print(f"\nAverage Character count before Data Preprocessing: {(processed_df['review'].str.len()).mean()}")
#remove the stop words
stop_words = set(stopwords.words('english'))
processed_df['review'] = processed_df['review'].apply(lambda text: ' '.join([word for word in str(text).split() if word.lower() not in stop_words]))

#perform lemmatization 
lemmatizer = WordNetLemmatizer()
processed_df['review'] = processed_df['review'].apply(lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split()]))

print(f"\nAverage Character count after Data Preprocessing: {(processed_df['review'].str.len()).mean()}\n\n")

random_reviews = processed_df.sample(n=3, random_state=36)

# Print the random reviews
print("\n\n")
for index, row in random_reviews.iterrows():
    print(f"Label: {row['label']}")
    print(f"Review: {row['review']}")
    print("\n" + "-"*50 + "\n") 



X_train, X_test, y_train, y_test = train_test_split(processed_df['review'], processed_df['label'], test_size=0.2, random_state=36)

# TF-IDF Feature Extraction
#Create a TfidfVectorizer to convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer() 

#Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

#ransform the test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#Now, X_train_tfidf and X_test_tfidf contain the TF-IDF features for training and testing sets

X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Perceptron
#Create and train the Perceptron model
perceptron_model = Perceptron(random_state=36)
perceptron_model.fit(X_train_tfidf, y_train)

#Make predictions on training and testing data
y_train_pred = perceptron_model.predict(X_train_tfidf)
y_test_pred = perceptron_model.predict(X_test_tfidf)

#Calculate metrics for training set
train_accuracy_Perceptron = accuracy_score(y_train, y_train_pred)
train_precision_Perceptron = precision_score(y_train, y_train_pred)
train_recall_Perceptron = recall_score(y_train, y_train_pred)
train_f1_Perceptron = f1_score(y_train, y_train_pred)

#Calculate metrics for testing set
test_accuracy_Perceptron = accuracy_score(y_test, y_test_pred)
test_precision_Perceptron = precision_score(y_test, y_test_pred)
test_recall_Perceptron = recall_score(y_test, y_test_pred)
test_f1_Perceptron = f1_score(y_test, y_test_pred)


# SVM

#Create and train the SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

#Make predictions on training and testing data
y_train_pred = svm_model.predict(X_train_tfidf)
y_test_pred = svm_model.predict(X_test_tfidf)

#Calculate metrics for training set
train_accuracy_SVM = accuracy_score(y_train, y_train_pred)
train_precision_SVM = precision_score(y_train, y_train_pred)
train_recall_SVM = recall_score(y_train, y_train_pred)
train_f1_SVM = f1_score(y_train, y_train_pred)

#Calculate metrics for testing set
test_accuracy_SVM = accuracy_score(y_test, y_test_pred)
test_precision_SVM = precision_score(y_test, y_test_pred)
test_recall_SVM = recall_score(y_test, y_test_pred)
test_f1_SVM = f1_score(y_test, y_test_pred)


# Logistic Regression

#Create and train the Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_tfidf, y_train)

#Make predictions on training and testing data
y_train_pred = logreg_model.predict(X_train_tfidf)
y_test_pred = logreg_model.predict(X_test_tfidf)

#Calculate metrics for training set
train_accuracy_logistic_regression = accuracy_score(y_train, y_train_pred)
train_precision_logistic_regression = precision_score(y_train, y_train_pred)
train_recall_logistic_regression = recall_score(y_train, y_train_pred)
train_f1_logistic_regression = f1_score(y_train, y_train_pred)

#Calculate metrics for testing set
test_accuracy_logistic_regression = accuracy_score(y_test, y_test_pred)
test_precision_logistic_regression = precision_score(y_test, y_test_pred)
test_recall_logistic_regression = recall_score(y_test, y_test_pred)
test_f1_logistic_regression = f1_score(y_test, y_test_pred)


# Naive Bayes

#Create and train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

#Make predictions on training and testing data
y_train_pred = nb_model.predict(X_train_tfidf)
y_test_pred = nb_model.predict(X_test_tfidf)

#Calculate metrics for training set
train_accuracy_naive_bayes = accuracy_score(y_train, y_train_pred)
train_precision_naive_bayes = precision_score(y_train, y_train_pred)
train_recall_naive_bayes = recall_score(y_train, y_train_pred)
train_f1_naive_bayes = f1_score(y_train, y_train_pred)

#Calculate metrics for testing set
test_accuracy_naive_bayes = accuracy_score(y_test, y_test_pred)
test_precision_naive_bayes = precision_score(y_test, y_test_pred)
test_recall_naive_bayes = recall_score(y_test, y_test_pred)
test_f1_naive_bayes = f1_score(y_test, y_test_pred)



print("Perceptron")
print(f"Training Accuracy: {train_accuracy_Perceptron}")
print(f"Training Precision: {train_precision_Perceptron}")
print(f"Training Recall: {train_recall_Perceptron}")
print(f"Training F1-score: {train_f1_Perceptron}")
print(f"Testing Accuracy: {test_accuracy_Perceptron}")
print(f"Testing Precision: {test_precision_Perceptron}")
print(f"Testing Recall: {test_recall_Perceptron}")
print(f"Testing F1-score: {test_f1_Perceptron}")

print("\nSVM")
print(f"Training Accuracy: {train_accuracy_SVM}")
print(f"Training Precision: {train_precision_SVM}")
print(f"Training Recall: {train_recall_SVM}")
print(f"Training F1-score: {train_f1_SVM}")
print(f"Testing Accuracy: {test_accuracy_SVM}")
print(f"Testing Precision: {test_precision_SVM}")
print(f"Testing Recall: {test_recall_SVM}")
print(f"Testing F1-score: {test_f1_SVM}")

print("\nLogistic Regression")

print(f"Training Accuracy: {train_accuracy_logistic_regression}")
print(f"Training Precision: {train_precision_logistic_regression}")
print(f"Training Recall: {train_recall_logistic_regression}")
print(f"Training F1-score: {train_f1_logistic_regression}")
print(f"Testing Accuracy: {test_accuracy_logistic_regression}")
print(f"Testing Precision: {test_precision_logistic_regression}")
print(f"Testing Recall: {test_recall_logistic_regression}")
print(f"Testing F1-score: {test_f1_logistic_regression}")

print("\nNaive Bayes")
print(f"Training Accuracy: {train_accuracy_naive_bayes}")
print(f"Training Precision: {train_precision_naive_bayes}")
print(f"Training Recall: {train_recall_naive_bayes}")
print(f"Training F1-score: {train_f1_naive_bayes}")
print(f"Testing Accuracy: {test_accuracy_naive_bayes}")
print(f"Testing Precision: {test_precision_naive_bayes}")
print(f"Testing Recall: {test_recall_naive_bayes}")
print(f"Testing F1-score: {test_f1_naive_bayes}")