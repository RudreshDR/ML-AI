from fastapi import FastAPI,Query
from model import  most_similar

app = FastAPI(title='Mini SSM NLP Module')

@app.get('/')
def root():
    return {"message":"mini ssm project  is live!"}

@app.get('/similar')
def get_similar(sentence: str = Query(..., description='Query sentence'), top_k: int = 3):
    return {"query":sentence,"results":most_similar(sentence,top_k)}