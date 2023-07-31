# Fake News Analysis using Machine Learning

## Overview
This project focuses on analyzing news articles to determine if they are real or fake. We use machine learning techniques to build models that can classify news articles into two classes: real news (class 1) and fake news (class 0). The main steps of the analysis include data preprocessing, feature extraction using TF-IDF, and training and evaluating different classifiers.

## Code Explanation

1. `data = pd.read_csv('News.csv', index_col=0)`: The code starts by loading the news data from the 'News.csv' file into a pandas DataFrame called `data`.

2. `data = data.drop(["title", "subject", "date"], axis=1)`: We drop unnecessary columns like 'title', 'subject', and 'date' from the DataFrame as they are not relevant for our analysis.

3. `data = data.sample(frac=1)`: To avoid any biases due to the order of data, we shuffle the DataFrame rows using `sample()`.

4. `import re, nltk`: We import necessary libraries for text preprocessing and tokenization.

5. `nltk.download('punkt')` and `nltk.download('stopwords')`: We download the NLTK data for tokenization and stop words.

6. `from nltk.corpus import stopwords`: We import the NLTK stop words corpus for removing common stop words from the text.

7. `from nltk.tokenize import word_tokenize`: We import the `word_tokenize` function for tokenizing sentences into words.

8. `from nltk.stem.porter import PorterStemmer`: We import the `PorterStemmer` class for word stemming.

9. `from wordcloud import WordCloud`: We import the `WordCloud` class for creating word cloud visualizations.

10. Preprocessing:
    - `preprocess_text(text_data)`: We define a function `preprocess_text` to clean the text data by removing punctuation, converting to lowercase, tokenizing, and removing stop words.

    - `preprocessed_review = preprocess_text(data['text'].values)`: We preprocess the 'text' column of the DataFrame `data` and store the cleaned text in a new list called `preprocessed_review`.

    - `data['text'] = preprocessed_review`: We replace the original 'text' column in the DataFrame `data` with the preprocessed text.

11. Word Cloud Visualization:
    - `consolidated = ' '.join(word for word in data['text'][data['class'] == 1].astype(str))`: We join all the text data labeled as class 1 (real news) into a single string called `consolidated`.

    - `wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False)`: We initialize a `WordCloud` object for visualization.

    - `plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')`: We generate and display the word cloud for real news.

12. Machine Learning:
    - `from sklearn.model_selection import train_test_split`: We import the `train_test_split` function for splitting data into training and testing sets.

    - `from sklearn.metrics import accuracy_score`: We import `accuracy_score` to evaluate the model's performance.

    - `from sklearn.linear_model import LogisticRegression`: We import `LogisticRegression` for one of the classification models.

    - `x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)`: We split the data into training and testing sets.

    - `from sklearn.feature_extraction.text import TfidfVectorizer`: We import `TfidfVectorizer` for text feature extraction.

    - `vectorization = TfidfVectorizer()`: We initialize a `TfidfVectorizer` object for TF-IDF vectorization.

    - `x_train = vectorization.fit_transform(x_train)`: We fit and transform the training data using the TF-IDF vectorizer.

    - `x_test = vectorization.transform(x_test)`: We transform the testing data using the same vectorizer.

    - Model Training and Evaluation:
        - Logistic Regression:
            - `model = LogisticRegression()`: We initialize a logistic regression model.

            - `model.fit(x_train, y_train)`: We train the logistic regression model on the training data.

            - `accuracy_score(y_train, model.predict(x_train))`: We compute the accuracy of the model on the training data.

            - `accuracy_score(y_test, model.predict(x_test))`: We compute the accuracy of the model on the testing data.

        - Decision Tree Classifier:
            - `from sklearn.tree import DecisionTreeClassifier`: We import `DecisionTreeClassifier` for another classification model.

            - `model = DecisionTreeClassifier()`: We initialize a decision tree classifier model.

            - `model.fit(x_train, y_train)`: We train the decision tree classifier model on the training data.

            - `accuracy_score(y_train, model.predict(x_train))`: We compute the accuracy of the model on the training data.

            - `accuracy_score(y_test, model.predict(x_test))`: We compute the accuracy of the model on the testing data.

        - Confusion Matrix:
            - `from sklearn import metrics`: We import `metrics` for calculating the confusion matrix.

            - `cm = metrics.confusion_matrix(y_test, model.predict(x_test))`: We compute the confusion matrix for the decision tree classifier.

            - `cm_display.plot()`: We plot the confusion matrix using `ConfusionMatrixDisplay`.

13. Conclusion:
    - The code demonstrates the process of text preprocessing, feature extraction using TF-IDF, and training and evaluating two classification models (Logistic Regression and Decision Tree Classifier) for fake news analysis.

    - The word cloud visualizations provide insights into the most frequently occurring words in real news and fake news.

    - The accuracy scores help in assessing the performance of the models on both the training and testing data.

    - The confusion matrix provides information about the model's classification performance for fake news and real news.

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

