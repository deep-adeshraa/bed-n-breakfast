import base64
import json
import boto3

SNS_ARN = "arn:aws:sns:us-east-1:107150785401:Customer-request"
sns_client = boto3.client('sns', region_name='us-east-1')


def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
    event (dict): Event payload.
    context (google.cloud.functions.Context): Metadata for the event.
    """

    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    data = json.loads(pubsub_message)
    print(data)
    user_id = data["user_id"]["S"]
    ticket_id = data["ticket_id"]["S"]
    price = data["price"]["N"]
    place = data["place"]["S"]
    start_date = data["start_date"]["S"]
    end_date = data["end_date"]["S"]

    msg = """
    New Tour booking request from customer.
    ------------------------------------------------------------------------------------
    {a:<20}    :   {user_id}
    {b:<20}    :   {ticket_id}
    {c:<20}    :   {price}
    {d:<20}    :   {place}
    {e:<20}    :   {start_date}
    {f:<20}    :   {end_date}
    ------------------------------------------------------------------------------------
    """.format(a='User', b='ticket id', c='Price', d="Place", e="Start date",
               f="End date", user_id=user_id, ticket_id=ticket_id,
               price=price, place=place, start_date=start_date,
               end_date=end_date)

    sns_client.publish(
        TopicArn=SNS_ARN,
        Message=msg,
        Subject="Hello! New Tour from customer"
    )

    print("Message sent")
