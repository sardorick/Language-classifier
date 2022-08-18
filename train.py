import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
np.random.seed(0)


# make a dictionary with models to test
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0, n_estimators=100),
    "Ada Boost": AdaBoostClassifier(random_state=0, n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0, n_estimators=100),
    "Logistic Regression": LogisticRegression(random_state=0)
    }

# load data
df = pd.read_csv("data.csv")
df.dropna(inplace=True)

# set vectorizers - both CountVectorizer and TFIDF Vectorizer to test both of them
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(df['file_body'])
tfvectorizer = TfidfVectorizer()
tfvectors = tfvectorizer.fit_transform(df['file_body'])

# split the data to training and test sets
x_train, x_test, y_train, y_test = train_test_split(tfvectors, df['language'], test_size=0.30, random_state=42)


# Results df
results = pd.DataFrame({'Model': [], "Accuracy Score": [], "Balanced Accuracy score": [], "Time": []})

for model_name, model in classifiers.items():
    start_time = time.time()

    model.fit(x_train, y_train)

    predics = model.predict(x_test)
    total_time = time.time() - start_time

    results = results.append({"Model": model_name,
                            "Accuracy Score": accuracy_score(y_test, predics)*100,
                            "Balanced Accuracy score": balanced_accuracy_score(y_test, predics)*100,
                            "Time": total_time}, ignore_index=True)

results_order = results.sort_values(by=['Accuracy Score'], ascending=False, ignore_index=True)
print(results_order)

"""
With count Vectorizer
                 Model  Accuracy Score  Balanced Accuracy score         Time
0          Extra Trees       86.334767                85.406084    67.129523
1        Random Forest       85.781602                84.782290    27.838532
2  Logistic Regression       84.572833                83.604689     7.862803
3    Gradient Boosting       83.200164                82.284173  1001.163250
4        Decision Tree       80.004098                79.214109     1.371663
5            Ada Boost       37.348904                34.861668    52.624517
"""
"""
With Tfidf Vectorizer
                 Model  Accuracy Score  Balanced Accuracy score         Time
0        Random Forest       86.908420                86.080146    26.568356
1          Extra Trees       86.826470                85.955004    65.820110
2  Logistic Regression       83.794304                81.765667     8.447470
3    Gradient Boosting       82.974800                81.934080  1071.766095
4        Decision Tree       79.102643                78.389962     1.587980
5            Ada Boost       46.609301                45.432870    55.026717
"""

# save the model to use in the terminal based app
import joblib
best_model = classifiers.get("Random Forest")
joblib.dump(best_model, 'model.pkl')

