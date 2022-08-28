var AWS = require('aws-sdk');
AWS.config.update({
  });


// Create the DynamoDB service object
var ddb = new AWS.DynamoDB({
    region: 'us-east-1',
    apiVersion: '2012-08-10'
});

//reference taken from - https://www.fernandomc.com/posts/eight-examples-of-fetching-data-from-dynamodb-with-node/
//function to get frequency of an existing item from DynamoDB
async function getOrders(params) {
    return new Promise(function(resolve, reject) {
        ddb.scan(params, function(err, data) {
            if (err) {
                console.log(err);
            } else {
                var orders = "{\"orders\":[";

                data.Items.forEach(function (element, index, array) {
                   var  orderCurrentStatus = "{\"orderCurrentStatus\":\"" + element.orderCurrentStatus.S+"\",";
                   var orderReadyTime =  "\"orderReadyTime\":\"" +element.orderReadyTime.S+"\",";
                   var total =  "\"total\":\"" +element.total.S+"\",";
                   var orderPlacedBy =  "\"orderPlacedBy\":\"" +element.orderPlacedBy.S+"\",";
                   var order =  "\"order\":" +element.order.S+",";
                   var orderDeliveredTime =  "\"orderDeliveredTime\":\"" +element.orderDeliveredTime.S+"\",";
                   var orderPlacedTime =  "\"orderPlacesTime\":\"" +element.orderPlacedTime.S+"\"";

                   var orderItem = orderCurrentStatus+orderReadyTime+total+orderPlacedBy+order+orderDeliveredTime+orderPlacedTime+"},";
                    orders = orders + orderItem;
                  });
                  orders= orders.slice(0,-1);
                resolve(orders+"]}");
            }
        })
    });
}

exports.getOrders = (req, res) => {

console.log(req.query.orderPlacedBy);
const params = {
    // Specify which items in the results are returned.
    FilterExpression: "orderPlacedBy  = :o",
    // Define the expression attribute value, which are substitutes for the values you want to compare.
    ExpressionAttributeValues: {
      ":o": {S: req.query.orderPlacedBy}
    },
    TableName: "orders",
  };

(async () => {
    var orders = await getOrders(params);
      res.setHeader("Content-Type", "application/json");
       res.setHeader('Access-Control-Allow-Origin', "*");
       res.setHeader('Access-Control-Allow-Methods', "*");
    res.status(200).send(orders);
})();

};
