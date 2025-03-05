# Text Category Classification Project

## Overview
This project focuses on classifying given text into specific categories using a Multinomial Naive Bayes classifier. The data is sourced from two different datasets, which are then processed and merged to create a unified dataset. The categories of interest are Business, Health, and Politics. We have used news data sourced from two different sources for the classification implementation.

## Data Sources
1. **UCI Machine Learning Repository**: [News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator)
   - Categories used: Business, Health
2. **Kaggle**: [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
   - Category used: Politics

## Data Processing
The `data_process.py` script handles the extraction and merging of data from the two sources. The resulting dataset contains the following columns:
- **Title**: The title of the news article.
- **URL**: The URL linking to the full article.
- **Category**: The category of the news article (Business, Health, or Politics).

## Classification
The `classification.ipynb` notebook contains the implementation for training the Multinomial Naive Bayes classifier. The following libraries and techniques are utilized:

### Natural Language Toolkit (NLTK)
- **Tokenization using `WhitespaceTokenizer`**: Tokenization is the process of splitting text into individual words or tokens. The `WhitespaceTokenizer` is used to split the text based on whitespace, which is a simple and effective way to prepare text for further processing.
- **Stopwords removal using `nltk.corpus.stopwords`**: Stopwords are common words that do not contribute much to the meaning of a sentence (e.g., "the", "is", "and"). Removing these words helps to reduce noise and improve the performance of the classifier.
- **Lemmatization using `WordNetLemmatizer`**: Lemmatization reduces words to their base or root form (e.g., "running" to "run"). This helps in normalizing the text and reducing the dimensionality of the feature space.

### Scikit-learn
- **Text feature extraction using `TfidfVectorizer`**: The `TfidfVectorizer` converts text into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This technique not only considers the frequency of words in a document but also their importance across the entire dataset, providing a more meaningful representation of the text.
- **Classification using `MultinomialNB`**: The Multinomial Naive Bayes classifier is a probabilistic model that is particularly well-suited for text classification tasks. It is efficient, easy to implement, and performs well with high-dimensional sparse data, such as text represented by TF-IDF features.

## Requirements
To run the scripts and notebook, ensure you have the following Python libraries installed:
    - `nltk`
    - `sklearn`
    - `pandas`
    - `numpy`

You can install the required libraries using pip:
```bash
pip install nltk scikit-learn pandas numpy
```

## Usage
1. **Data Processing**:
   - Manually download the files from above mentioned source and place them under folder_name dataset with filename saved as data.csv and data.json
   - Run `data_process.py` to download and merge the datasets.
   ```bash
   python data_process.py
   ```

2. **Classification**:
   - Open and run the `classification.ipynb` notebook to train and evaluate the classifier.
   ```bash
   jupyter notebook classification.ipynb
   ```

## Summary
This project demonstrates the process of data collection, preprocessing, and classification of news articles into predefined categories using a Multinomial Naive Bayes model. The integration of data from multiple sources and the application of NLP techniques ensure a robust classification system.

## References
   - Gasparetti, F. (2017). News Aggregator [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F61C.
   - Misra, R. (2022). News Category Dataset. arXiv Preprint arXiv:2209. 11429.
   - Misra, R., & Grover, J. (01 2021). Sculpting Data for ML: The first act of Machine Learning.