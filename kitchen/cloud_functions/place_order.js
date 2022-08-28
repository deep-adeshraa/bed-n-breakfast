const express = require("express");
const app = express();
const axios = require('axios');
var AWS = require('aws-sdk');
var easyinvoice = require('easyinvoice');
const { v4: uuidv4 } = require('uuid');
//var fs = require('fs');
const cors = require('cors');
const {PubSub} = require('@google-cloud/pubsub');
const pubSubClient = new PubSub("ultra-mediator-352903");
AWS.config.update({
  });

// Create the DynamoDB service object
var ddb = new AWS.DynamoDB({
    region: 'us-east-1',
    apiVersion: '2012-08-10'
});
var s3 = new AWS.S3({
    params: {
        Bucket: 'invoices-dalsoft5410'
    }
});

async function delayTillReady(timedelay,orderId,orderPlaceBy) {
  const dataBuffer = Buffer.from(timedelay+','+orderId+','+orderPlaceBy);

  try {
    const messageId = await pubSubClient
      .topic('projects/ultra-mediator-352903/topics/orderStatus')
      .publishMessage({data: dataBuffer});
    console.log(`Message ${messageId} published.`);
  } catch (error) {
    console.error(`Received error while publishing: ${error.message}`);
  }
}

// async function delayTillReady(timedelay,orderId,orderPlaceBy) {
//   const dataBuffer = Buffer.from(timedelay+','+orderId+','+orderPlaceBy);
// //https://us-central1-ultra-mediator-352903.cloudfunctions.net/delay
// // const options = {
// // 	url: "https://us-central1-ultra-mediator-352903.cloudfunctions.net/delay",
// // 	port: "80",
// // 	method: "POST",
// //     data: dataBuffer
// // }

// // const request = http.request(options, (response) => {
// // 	console.log(response);
// // });
// axios
//   .post('https://us-central1-ultra-mediator-352903.cloudfunctions.net/delay', {
//     message: dataBuffer,
//   })
//   .then(res => {
//     console.log(`statusCode: ${res.status}`);
//     console.log(res);
//   })
//   .catch(error => {
//     console.error(error);
//   });
// //   try {
// //     const messageId = await pubSubClient
// //       .topic('projects/ultra-mediator-352903/topics/orderStatus')
// //       .publishMessage({data: dataBuffer});
// //     console.log(`Message ${messageId} published.`);
// //   } catch (error) {
// //     console.error(`Received error while publishing: ${error.message}`);
// //   }
// }
//function to create file in bucket
async function createFile(data) {
    return new Promise(function(resolve, reject) {
        s3.putObject(data, function(err, data) {
            if (err) {
                console.log(err);
                resolve(false);
            } else {
                resolve(true);

            }
        });
    });
}
async function generateInvoice(orderItems,orderPlaceBy,invoiceNumber,firstName,lastName){
    console.log("orderItems"+JSON.stringify(orderItems));
    var taxrate = 15;
    var orderitem ="[";
    for(let i=0;i<orderItems.length;i++){
        orderitem = orderitem + '{"quantity":"'+orderItems[i].quantity+'","description":"'+orderItems[i].name+'","tax-rate":"'+taxrate+'","price":"'+orderItems[i].price+'"},';
    }
    orderitem = orderitem.slice(0,-1);
    orderitem = orderitem+"]";
    console.log("orderietm"+orderitem);

    var orderItems = "";
    var emailId = orderPlaceBy;
    // var firstName = "Manali";
    // var lastName = "Shah";
    var data = {
        "client": {
         "company": emailId,
         "zip": lastName,
          "city": firstName
     },
         "sender": {
             "company": "DALSoft5410",
             "address": "Goldberg Computer Science Building",
             "zip": "B3H 1W5",
             "city": "Halifax",
             "country": "Canada"
         },
         "information": {
             "number": invoiceNumber,
             "date": new Date().toISOString().slice(0, 10)
         },

         "products": JSON.parse(orderitem),

         "settings": {
             "currency": "CAD",
         },
     };

     easyinvoice.createInvoice(data, function (result) {
        (async () => {
         var pdfBuffer = Buffer.from(result.pdf, 'base64')
             var data = {
                 Key: invoiceNumber,
                 Body: pdfBuffer,
                 ContentEncoding: 'base64',
                 ContentType: 'application/pdf'
             };
         await createFile(data);
     })();

     });

}

//reference taken from - https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_UpdateItem.html
//function to put item to DynamoDB
async function putItemToDB(putparams) {
    return new Promise(function(resolve, reject) {
        ddb.putItem(putparams, function(err, data) {
            if (err) {
                //resolve true if the condition expression fail
                console.log(err);
                resolve(false);
            } else {
                console.log(data);
                //resolve false if item is successfully added to DynamoDB
                resolve(true);
            }
        })
    });
}

app.use(cors());

app.post("/", (req, res) => {

    var now = new Date();
    var maxtime =0;
    for(let i=0;i<req.body.orderItems.length;i++) {
        if(req.body.orderItems[i].preparationTime>maxtime){
            maxtime = req.body.orderItems[i].preparationTime;
        }
    }
    var minutesToAdd=maxtime;
    var currentDate = new Date();
    var readyTime = new Date(currentDate.getTime() + minutesToAdd*60000);
    var orderId = uuidv4();
    var invoiceNumber=orderId.split('-')[0];
    var isSuccess = false;
    var firstName = req.body.firstName;
    var lastName = req.body.lastName;
    var params = {
        TableName: 'orders',
        Item: {
            'id': {
                S: orderId
            },
            'order': {
                S: JSON.stringify(req.body.orderItems)
            },
            'orderPlacedTime': {
                S: now.toISOString()
            },
            'orderReadyTime': {
                S: readyTime.toISOString()
            },
            'orderDeliveredTime': {
                S: ''
            },
            'orderCurrentStatus': {
                S: 'Order Placed'
            },
            'orderPlacedBy': {
                S: req.body.orderPlaceBy
            },
            'total': {
                S: String(req.body.total)
            },
            'invoiceNumber' :{
                S: invoiceNumber
            }
        }
    };
    try {
        (async () => {
            isSuccess = await putItemToDB(params); //put item to DynamoDB with updated frequency value
            await generateInvoice(req.body.orderItems,req.body.orderPlaceBy,invoiceNumber,firstName,lastName);
            console.log(req.body.orderPlaceBy);
            await delayTillReady(minutesToAdd,orderId,req.body.orderPlaceBy);
            if(isSuccess) {
                    res.send(JSON.parse( "{\"response\":\"Order Placed Successfully!\"}"));

            } else {
                res.send(JSON.parse("{\"response\":\"There was some error while placing the order. Please try again after sometime.\"}"));
            }
        })();

    }catch (error) {
        console.error(error);

    }
});

exports.placeOrder = app;

