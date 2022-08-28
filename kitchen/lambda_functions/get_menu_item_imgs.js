const AWS = require('aws-sdk');
const s3 = new AWS.S3();
const dynamo = new AWS.DynamoDB.DocumentClient();

async function geturl (id) {
     return new Promise(function(resolve, reject) {
     console.log("getUrl");
    const myBucket = 'menu-item-img-dalsoft5410'
    const myKey = id;
    const signedUrlExpireSeconds = 60 * 5
    const url = s3.getSignedUrl('getObject', {
        Bucket: myBucket,
        Key: myKey,
        Expires: signedUrlExpireSeconds
    })
    
    resolve(url);
     });
     
}
exports.handler = async (event, context) => {
    //console.log('Received event:', JSON.stringify(event, null, 2));

     let statusCode = '200';
    const headers = {
        'Content-Type': 'application/json',
        'access-control-allow-headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'access-control-allow-methods': 'OPTIONS,POST',
        'access-control-allow-origin': '*'
    };
    var id =event.queryStringParameters.id;
    var body =  await geturl(id);
    body = '{"url":"'+body+'"}'
    return {
        statusCode,
        body,
        headers,
    };
};
