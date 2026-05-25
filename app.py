Step 1: Import Libraries
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')
from nltk.corpus import stopwords

step 2:Load Dataset
df = pd.read_csv("dataset.csv")

Step 3: Select Required Columns
df = df[['Text', 'Score']]

Step 4: Convert Ratings into Sentiment
def convert_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"
df['Sentiment'] = df['Score'].apply(convert_sentiment)

Step 5: Clean Text
stop_words = set(stopwords.words('english'))
def clean_text(text):
     text = str(text).lower()
     text = re.sub(r'http\\S+', '', text)
     text = re.sub(r'[^a-zA-Z ]', '', text)
     words = text.split()
     words = [word for word in words if word not in stop_words]
     return " ".join(words)
     df['Cleaned_Text'] = df['Text'].apply(clean_text)
 Step 6: Feature Extraction using TF-IDF
   tfidf = TfidfVectorizer(max_features=5000)
   X = tfidf.fit_transform(df['Cleaned_Text'])
   y = df['Sentiment']
Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
Step 8: Train Model
     model = LinearSVC()
     model.fit(X_train, y_train)
Step 9: Predict Results
     y_pred = model.predict(X_test)
Step 10: Check Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\\nClassification Report:\\n")
    print(classification_report(y_test, y_pred))
Step 11: User Input Prediction
   review = input("\\nEnter Amazon Review: ")
   clean_review = clean_text(review)
   review_vector = tfidf.transform([clean_review])
   prediction = model.predict(review_vector)
   print("Predicted Sentiment:", prediction[0])


