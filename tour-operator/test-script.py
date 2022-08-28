from urllib.request import Request
import flask
import urllib3
import uvicorn
import requests
import boto3
from collections import OrderedDict


from fastapi import FastAPI, Request

app = flask.Flask(__name__)

from typing import Dict
from flask import Flask, jsonify

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from numpy import place
from google.protobuf.json_format import MessageToJson


def addCors(response, code=200):
    headers = {'Access-Control-Allow-Origin': '*'}
    return (response, code, headers)

def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        return prediction


def predict_price(days, place):
    data = predict_tabular_classification_sample(
        project="",
        endpoint_id="",
        location="us-central1",
        instance_dict={"Days": str(days), "City": place}
    )

    res = {}
    res['classes'] = []
    res['scores'] = []

    for i in data['classes']:
        res['classes'].append(int(i))

    for i in data['scores']:
        res['scores'].append(i)

    return res

@app.route('/', methods=['GET'])
def deletefile():
    data = predict_price("3", "Halifax")
    return jsonify(data)



if __name__ == "__main__":
    app.run()

