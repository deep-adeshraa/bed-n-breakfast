const AWS = require("aws-sdk");
const dynamo = new AWS.DynamoDB({apiVersion: '2012-08-10'});

exports.handler = async (event) => {
    var params = {
      TableName: 'RoomDetails',
      Key: {
            id: {S: event.queryStringParameters.id}
      }
    };
    var result = await dynamo.deleteItem(params).promise();
    result = JSON.stringify(result);
   return result;
};