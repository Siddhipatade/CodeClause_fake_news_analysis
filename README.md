# Fake News Analysis using Machine Learning

## Overview
This project focuses on analyzing news articles to determine if they are real or fake. I use machine learning techniques to build models that can classify news articles into two classes: real news (class 1) and fake news (class 0). The main steps of the analysis include data preprocessing, feature extraction using TF-IDF, and training and evaluating different classifiers.

## Steps -
1. Importing Libraries and Datasets
2. Data Preprocessing
3. Preprocessing and analysis of the News column
4. Converting text into Vectors
5. Model training, Evaluation, and Prediction

# Code Explanation
### Preprocessing Text Data
The code first preprocesses the text data to clean and prepare it for analysis. The preprocessing steps include:

- Removing punctuation from the text.
- Converting the text to lowercase.
- Tokenizing the text into individual words.
- Removing stop words (commonly occurring words like "the," "is," "and," etc.).
- The preprocessed text data is stored back in the DataFrame.

### Visualizing Word Frequencies
The code generates a word cloud visualization to display the most frequently occurring words in both real and fake news articles. Word clouds help visualize the importance of words in the text data, with larger fonts indicating higher word frequencies.

### Building Machine Learning Models
The code splits the preprocessed text data into training and testing sets. Two machine learning models are used for classification:

Logistic Regression: A linear model commonly used for binary classification tasks.
Decision Tree Classifier: A non-linear model that partitions the feature space to make predictions.
Both models are trained on the training data and then evaluated on the testing data using accuracy scores.

### Evaluating Model Performance
The code computes accuracy scores for both models to evaluate their performance on the training and testing data. Accuracy is a metric that measures the percentage of correct predictions.

Additionally, for the Decision Tree Classifier, a confusion matrix is generated to gain insights into its classification performance for each class (real and fake) in the test data.

## Conclusion
The code demonstrates a text classification approach for fake news analysis. It preprocesses the text data, visualizes word frequencies, and builds machine learning models to predict whether a news article is real or fake. The models' performance is evaluated using accuracy scores, and insights into classification performance are gained using a confusion matrix.


## Requirements
- pandas
- seaborn
- matplotlib
- tqdm
- nltk
- wordcloud
- scikit-learn

## How to Use
1. Install the required packages using `pip install pandas seaborn matplotlib tqdm nltk wordcloud scikit-learn`.

2. Place the 'News.csv' file containing the text data and 'class' labels in the same directory as the script.

3. Run the script to perform the fake news analysis, train the models, and visualize the results.

## Acknowledgments
This project uses the 'News.csv' dataset, which contains text data and class labels for real and fake news articles.

