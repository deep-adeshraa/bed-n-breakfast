var AWS = require("aws-sdk");
const s3 = new AWS.S3();
var body = "";
AWS.config.update({region: "us-east-1"});

var dynamodb = new AWS.DynamoDB({apiVersion: "2012-08-10"});


async function scanForInvoiceUrls(id){
    try {
        var urls="";
        var params = {
            TableName: "orders",
            FilterExpression: "orderPlacedBy = :o",
      ExpressionAttributeValues: {
        ":o": { S: id }
      },
        };
        var result = await dynamodb.scan(params).promise()
        console.log(result.Items.length);
        for(let i=0;i<result.Items.length;i++) {
             urls = urls + '"'+await geturl(result.Items[i].invoiceNumber.S) +'",';
        }
        
       urls = "{\"invoices\":["+urls.slice(0,-1)+"]}"
       console.log(urls)
        return urls;
    } catch (error) {
        console.error(error);
    }
}

async function geturl (id) {
     return new Promise(function(resolve, reject) {
     console.log("getUrl");
    const myBucket = 'invoices-dalsoft5410'
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
exports.handler = async (event) => {
    let statusCode = '200';
    const headers = {
        'Content-Type': 'application/json',
        'access-control-allow-headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'access-control-allow-methods': 'OPTIONS,POST',
        'access-control-allow-origin': '*'
    };
   
var id =event.queryStringParameters.id
   var body =  await scanForInvoiceUrls(id);
    return {
        statusCode,
        body,
        headers,
    };
};
