from fastapi import FastAPI

import pickle
import string
import nltk
import uvicorn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pydantic import BaseModel

ps = PorterStemmer()

app = FastAPI()

class InputData(BaseModel):
    message: str

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.get('/')
def home():
    return {'Title':'Welcome to Email Spam Classification!'}

@app.post("/")
def predict_spam(input_data: InputData):
    #preprocess
    transformed_sms = text_transform(input_data.message)
    #vectorize
    vector_input =  tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result == 1:
        return {"result": "SPAM"}
    else:
        return {"result": "NOT SPAM"}

if __name__ == '__main__':
    uvicorn.run(app)