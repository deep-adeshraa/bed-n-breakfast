const AWS = require("aws-sdk");

const dynamo = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event, context) => {
  let body;
  let statusCode = 200;
  const headers = {
    "Content-Type": "application/json"
  };

  try {
    switch (event.routeKey) {
      case "DELETE /feedback/{id}":
        await dynamo
          .delete({
            TableName: "feedback",
            Key: {
              id: event.pathParameters.id
            }
          })
          .promise();
        body = `Deleted item ${event.pathParameters.id}`;
        break;
      case "GET /feedback/{id}":
        body = await dynamo
          .get({
            TableName: "feedback",
            Key: {
              id: event.pathParameters.id
            }
          })
          .promise();
        break;
      case "GET /feedback":
        body = await dynamo.scan({ TableName: "feedback" }).promise();
       body = body.Items;
        break;
      case "PUT /feedback":
        let requestJSON = JSON.parse(event.body);
        await dynamo
          .put({
            TableName: "feedback",
            Item: {
              id: requestJSON.id,
              email: requestJSON.email,
              name: requestJSON.name,
              feedback:requestJSON.feedback,
              sentiment:requestJSON.sentiment,
            }
          })
          .promise();
        body = `Put item ${requestJSON.id}`;
        break;
      default:
        throw new Error(`Unsupported route: "${event.routeKey}"`);
    }
  } catch (err) {
    statusCode = 400;
    body = err.message;
  } finally {
    body = JSON.stringify(body);
  }

  return {
    statusCode,
    body,
    headers
  };
};