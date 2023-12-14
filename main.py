from fastapi import FastAPI, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import Optional
from Service.sentimental_analysis_model import inference_model

app = FastAPI(docs_url='/', title='Sentiment_Analysis_Service',
              description='This is the Service used to make inference on model to predict the sentiment of comment'
              , version='1.0.0.1')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware('http')
async def count_request_handle_time(request: Request, handle_call):
    start_time = time.time()
    response = await handle_call(request)
    end_time = time.time() - start_time
    response.headers['X-Process-Time'] = str(end_time)
    return response


@app.post('/Service/sentimental_analysis')
async def post_sentimental_service(input_sequence: str = Body(embed=True)):
    predicted_index = inference_model(input_sequence)
    labels = ['Negative', 'Positive']
    response_ob = {}
    if predicted_index != -1:
        response_ob = {
            'status': True,
            'message': 'Analyze successfully data',
            'data': labels[predicted_index]
        }
    else:
        response_ob = {
            'status': False,
            'message': 'Failed to analyze data',
            'data': ''
        }
    return response_ob
