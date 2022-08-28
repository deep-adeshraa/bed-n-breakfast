const AWS = require("aws-sdk");
const dynamo = new AWS.DynamoDB({apiVersion: '2012-08-10'});

exports.handler = async (event) => {
     let requestJSON = JSON.parse(event.body);
      var params = {
      TableName: 'RoomBookings',
      Item: {
        'id' :{S:requestJSON.id},
        'toDate' :{S: requestJSON.toDate},
        'fromDate' :{S:requestJSON.fromDate},
        'noBeds' :{S:requestJSON.noBeds},
        'roomPrice' :{S: requestJSON.roomPrice},
        'roomId':{S:requestJSON.roomId },
        'bookingdatetime':{S:`${new Date().toISOString()}`},
        'customerId':{S:requestJSON.customerId },
      }
    };
    var result = await dynamo.putItem(params).promise();
    result = JSON.stringify(result);
    return result;
};