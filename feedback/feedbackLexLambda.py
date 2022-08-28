import json
import boto3
import urllib3
import base64
import random
client = boto3.client('dynamodb')
result = {}
intent=""
userid=""
dishid =""
quantity=""
res=""

def lambda_handler(event, context):
    intent=(event['currentIntent']['name'])
    print('the intent is : ',intent)
    if intent=='feedback':
        name=(event['currentIntent']['slotDetails']['name']['originalValue'])
        email=(event['currentIntent']['slotDetails']['email']['originalValue'])
        feedbacks=(event['currentIntent']['slotDetails']['feedbacks']['originalValue'])
        print("name: ",name)
        print("Email: ",email)
        print("feedback: ",feedbacks)
       
        feedback(name,email,feedbacks)
        return result


def feedback(name,email,feedbacks):
        url="https://us-central1-central-archery-275005.cloudfunctions.net/sentimentanalysis"
        data={
              "feedback": feedbacks
        }
        
        param = json.dumps(data)
        http = urllib3.PoolManager()
        res = http.request('POST', url, headers={'Content-Type': 'application/json'}, body=param)
        final = float(res.data.decode('utf-8'))
     
        print (final)
        if(final>0):
            feedtype="Positive"
        elif(final<1):
            feedtype="Negative"
        else:
            feedtype="Neutral"
        print(feedtype)
        
        form_data = {

                    "id": str(random.randint(100, 99999)),
                    "name": name,
                    "email": email,
                    "feedback": feedbacks,
                    "sentiment": feedtype,
                };
        print("heya ",form_data)
        
        url1="https://4ucj1xcie9.execute-api.us-east-1.amazonaws.com/feedback"
        
        param1 = json.dumps(form_data)
        http1 = urllib3.PoolManager()
        res1 = http1.request('PUT', url1, headers={'Content-Type': 'application/json'}, body=param1)
        final1 = res1.data.decode('utf-8')
        print("put api is ",final1) 
        
        global result
    
        result ={
             
            "dialogAction": {
        "type": "Close",
        "fulfillmentState": "Fulfilled",
        "message": {
            "contentType": "PlainText",
            "content": "The Feedback has been successfully submitted. Thank you!"
        },
         }
            }
                
        return result

