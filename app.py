import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


# load the test data to fit the TFIDF Vectorizer
df = pd.read_csv("data.csv")
df.dropna(inplace=True)

tfvectorizer = TfidfVectorizer()
tfvectors = tfvectorizer.fit_transform(df['file_body'])

# Load the trained model using joblib
model = joblib.load('model.pkl')

# parse the argument to make a terminal based app
parser = argparse.ArgumentParser(description = 'Programming language predictor')
parser.add_argument('test_data', type=str, help ='Input path to test data')
args = parser.parse_args()

test_data_link = args.test_data

# predict function
def predict(data):
    df = pd.read_csv(data)
    x_vectorized = tfvectorizer.transform(df['file_body'])
    pred = model.predict(x_vectorized)
    if len(df) != 1:
        for i in range(len(pred)):
            print(f"The programming language in sample number {i+1} is {pred[i]}.")
    else:
        print("The programming language in the file is", pred[0])

predict(test_data_link)
