from datetime import date, datetime
import json
import boto3
import urllib3
import base64
client = boto3.client('dynamodb')

http = urllib3.PoolManager()


def lambda_handler(event, context):
    # intent = (event['sessionState']['intent']['name'])

    return book_tour(event)


def book_tour(event):

    user_email = (event['currentIntent']['slots']['user_email'])
    from_date = (event['currentIntent']['slots']['from_date'])
    To_date = (event['currentIntent']['slots']['To_date'])
    place = (event['currentIntent']['slots']['place'])

    try:
        from_date_obj = datetime.strptime(from_date, "%Y-%m-%d")
        to_date_obj = datetime.strptime(To_date, "%Y-%m-%d")
        days = (to_date_obj - from_date_obj).days

        predict_data = http.request(
            'GET', url=f"https://us-central1-peak-service-312506.cloudfunctions.net/predict-prices?place={place}&days={days}", headers={'Content-Type': 'application/json'})

        predict_data = predict_data.data
        scores = predict_data["scores"]
        index = scores[index(max(scores))]
        price = predict_data["price"][index]

        book_ticket = http.request(
            "GET", url=f"https://us-central1-peak-service-312506.cloudfunctions.net/create-ticket?place={place}&start_date={from_date}&end_date={To_date}&price={price}&user_id={user_email}")

        return {
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": "Fulfilled",
                "message": {
                    "contentType": "PlainText",
                    "content": "The Tour Booked Successfully. Thank you!"
                },
            }
        }

    except Exception as e:
        print(e)
        return {
            "dialogAction": {
                "type": "Close",
                "fulfillmentState": "Fulfilled",
                "message": {
                    "contentType": "PlainText",
                    "content": "Something went wrong. Please proceed again!"
                },
            }
        }
