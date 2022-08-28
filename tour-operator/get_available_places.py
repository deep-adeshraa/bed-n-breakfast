import boto3
from flask import jsonify

dynamodb_client = boto3.client('dynamodb',
                               region_name='us-east-1')

TOUR_DETAILS_TABLE_NAME = 'tour-details'


def add_cors(response, code=200):
    headers = {'Access-Control-Allow-Origin': '*'}
    return (response, code, headers)


def scan_table():
    """
    Generates all the items in a DynamoDB table.
    """

    paginator = dynamodb_client.get_paginator("scan")

    for page in paginator.paginate(TableName=TOUR_DETAILS_TABLE_NAME):
        yield from page["Items"]


def function_handle(request):
    request_json = request.get_json()
    print(request_json)
    data = []

    for item in scan_table():
        data.append({
            "place": item["place"]["S"],
            "description": item["description"]["S"],
            "image": item["image"]["S"]
        })

    return add_cors(jsonify({
        "success": True,
        "data": data
    }))
