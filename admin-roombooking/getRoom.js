const AWS = require("aws-sdk");
const dynamo = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
   let body = await dynamo.scan({ TableName: "RoomDetails" }).promise();
   body = JSON.stringify(body.Items);
   return body;
};