import boto3
from boto3.dynamodb.conditions import Key
from flask import jsonify


dynamodb_client = boto3.resource('dynamodb',
                               region_name='us-east-1')
                               
TOUR_DETAILS_TABLE_NAME = "tour-details"
TOUR_TICKETS_TABLE_NAME = "tour-tickets"


def addCors(response, code=200):
    headers = {'Access-Control-Allow-Origin': '*'}
    return (response, code, headers)


def function_handle(request):
    request_json = request
    user_id = request_json.args.get('user_id')

    if not user_id:
        return addCors(jsonify({
            "success": False
        }))

    table = dynamodb_client.Table(TOUR_TICKETS_TABLE_NAME)
    response = table.scan(
        FilterExpression=Key('user_id').eq(user_id),
    )

    return addCors(jsonify({"tickets": response['Items']}))
