'use strict';

const aws = require('aws-sdk');
var uuid = require('uuid');
const moment = require("moment");


exports.handler = async (event, context) => {
  const docClient = new aws.DynamoDB.DocumentClient();
     let requestedService = event.currentIntent.slots.menu;
 
    if(requestedService === 'menu'){
   	const params = {"TableName": "menuItems", "ProjectionExpression" : "ItemName"}
    const data = await docClient.scan(params).promise()
    const stringData = JSON.stringify(data.Items);
    var menu = event.currentIntent.slots.menu;
    
    return {
					"dialogAction": 
					{
							"type": "Close",
							"fulfillmentState": "Fulfilled",
							"message": {
							"contentType": "PlainText",
							"content": "Here is our Breakfast menu"+stringData,
							}
							
					}
					}
   	
   }
   else
   {
 	
   	const docClient = new aws.DynamoDB.DocumentClient();
   	
   	const orderPlacedTime =moment().format("YYYY-DD-MM: HH:mm:ss")
	const orderReadyTime = moment().add(20, 'minutes').format("YYYY-DD-MM: HH:mm:ss");
	var itemName = event.currentIntent.slots.itemName;
	var quantity = event.currentIntent.slots.quantity;
	var emailAdd = event.currentIntent.slots.emailAdd;
	
	var p1 = 10.55, p2=30.55, p3=5.99,p4=5.99,p5=4.99,p6=20.99,total=0 ;
	
	if(itemName === 'French Bread'){
	    total = quantity*p1;
    
	}
	else if(itemName === 'Pizza'){
	    total = quantity*p2;
	}
	else if(itemName === 'Cheese Burger'){
	    total = quantity*p3;
	}
	else if(itemName === 'Fruit Dish'){
	    total = quantity*p4;
	}
	else if(itemName === 'Coffee'){
	    total = quantity*p5;
	}
	else {
	    total = quantity*p6;
	}
	
	    
    await docClient.put({TableName : 'orders',Item: {id:uuid.v1(), order:itemName, orderPlacedBy: emailAdd,orderPlacedTime :orderPlacedTime,orderReadyTime:orderReadyTime,total:total}}).promise();
	
	return {
					"dialogAction": 
					{
							"type": "Close",
							"fulfillmentState": "Fulfilled",
							"message": {
							"contentType": "PlainText",
							"content": "Your order for "+quantity+"  "+itemName+ " is placed and the bill is "+total+"$, your order will be ready by --> "+orderReadyTime,
							}
							
					}
					}
   	
   	
   	
   }
	

};