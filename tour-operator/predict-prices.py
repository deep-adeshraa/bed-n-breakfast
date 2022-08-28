# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_tabular_classification_sample]

# Requirements
# numpy
# protobuf
# gcloud
# google-cloud-aiplatform

from typing import Dict
from flask import jsonify

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from numpy import place


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
        return dict(prediction)

    # return predictions


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


def function_handle(request):
    days = request.args.get('days')
    place = request.args.get('place')

    try:
        data = predict_price(days, place)

        return addCors(jsonify({
            "success": True,
            "data": dict(data)
        }))
    except:
        return addCors(jsonify({
            "success": False,
            "data": "Invalid details"
        }))
