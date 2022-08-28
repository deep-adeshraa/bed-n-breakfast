const AWS = require("aws-sdk");
const dynamo = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
   let body = await dynamo.scan({ TableName: "RoomBookings" }).promise();
   body = JSON.stringify(body.Items);
   return body;
};
