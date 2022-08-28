const https = require('https');
const AWS = require('aws-sdk');
const dynamo = new AWS.DynamoDB.DocumentClient({
  region: "ca-central-1",
});

var getCustomerID= async (email) => {
   let body = await dynamo.scan({ TableName: "serverless-project-users" }).promise();
    body = JSON.stringify(body.Items);
    var customerList=JSON.parse(body);
    var i = customerList.length;
    while ( i --> 0 ) {
        if(customerList[i].email==email)
        {
            return customerList[i].uid;
        }
    }
    return " "
}

var getTotalDays = (fromDate,toDate) => {
        var date1 = new Date(fromDate);
        var date2 = new Date(toDate);              
        return (parseInt((date2 - date1) / (1000 * 60 * 60 * 24)));
}

var addBookingRequest = async (url,data) =>{
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length,
    },
  }
  
  return new Promise((resolve, reject) => {
    const req = https.request(url, options, (res) => {
      if (res.statusCode < 200 || res.statusCode > 299) {
        return reject(new Error(`HTTP status code ${res.statusCode}`))
      }
      const body = []
      res.on('data', (chunk) => body.push(chunk))
      res.on('end', () => {
        const resString = Buffer.concat(body).toString()
        resolve(resString)
      })
    })
    req.on('error', (err) => {
      reject(err)
    })
    req.on('timeout', () => {
      req.destroy()
      reject(new Error('Request time out'))
    })
    req.write(data)
    req.end()
  })
}

var getData = async (URL) => {
     let dataString = '';
        await new Promise((resolve, reject) => {
        const req = https.get(URL, function(res) {
              res.on('data', chunk => {
                dataString += chunk;
              });
              res.on('end', () => {
                  resolve(dataString);
              });
            });
        });
        var roomData=JSON.parse(dataString);
      return roomData;
};
exports.handler = async (event, contex) => {
    let requestJSON = event.currentIntent.slots;
    var roomList= await getData("https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/getRoom")
    var bookingList= await getData("https://ni4d251mw1.execute-api.us-east-1.amazonaws.com/getRoomBooking")
    console.log(requestJSON);
    if(requestJSON.roomNo===null)
    {
        var availableRooms=[];
        var i = roomList.length;
        roomList.forEach(function (room) {
            room.isAvailable = true;
        });
        while ( i --> 0 ) {
            if(roomList[i].noBeds >= requestJSON.noBeds)
            {
                availableRooms.push(roomList[i]);
            }
        }
        var j = availableRooms.length;
        while ( j --> 0 ) {

                var roomID=availableRooms[j].id;
                var k = bookingList.length;
                while ( k --> 0 ) {
                    if(bookingList[k].roomId===roomID&&bookingList[k].fromDate===requestJSON.fromDate&&bookingList[k].toDate===requestJSON.toDate)
                    {
                        availableRooms[j].isAvailable=false;
                        break;
                    }
                    else if(bookingList[k].roomId!==roomID&&bookingList[k].fromDate!==requestJSON.fromDate&&bookingList[k].toDate!==requestJSON.toDate)
                    {
                        availableRooms[j].isAvailable=true;
                    }
                }
        }
        var displayString="Available Rooms...............";
        availableRooms.forEach(function (room) {
           displayString=displayString+"Room No: "+room.roomNo+"\nRoom Price: "+room.roomPrice+" ................";
        });
        displayString=displayString+"Please select the Room Number";
        return {"dialogAction": {
        "type": "Close",
        "fulfillmentState": "Fulfilled",
        "message": {
            "contentType": "PlainText",
            "content": displayString
        },
         }}
    }
    else if(requestJSON.roomNo!==null)
    {
        var ID=await getCustomerID(requestJSON.email);
        if(ID===" ")
        {
            return {"dialogAction": {
            "type": "Close",
            "fulfillmentState": "Fulfilled",
            "message": {
                "contentType": "PlainText",
                "content": "Invalid User!"
            },
             }}
        }
        var roomPrice=0;
        var roomID=0;
        roomList.forEach(function (room) {
            if(requestJSON.roomNo===room.roomNo)
            {
                roomPrice=room.roomPrice;
                roomID=room.id;
            }
        });
        var bookingTotal=bookingList.length + 10;
        var totalRoomPrice=getTotalDays(requestJSON.fromDate,requestJSON.toDate) * roomPrice;
        const data = JSON.stringify({
              "id" :bookingTotal.toString(),
              "toDate" :requestJSON.toDate,
              "fromDate" :requestJSON.fromDate,
              "noBeds" :requestJSON.noBeds,
              "roomPrice" :totalRoomPrice.toString(),
              "roomId":roomID,
              "customerId":ID.toString()
        });
       const res = await addBookingRequest('https://ni4d251mw1.execute-api.us-east-1.amazonaws.com/addRoomBooking', data)
        return {"dialogAction": {
        "type": "Close",
        "fulfillmentState": "Fulfilled",
        "message": {
            "contentType": "PlainText",
            "content": "The Room Booked Successfully. Thank you!"
        },
         }}
    }
};