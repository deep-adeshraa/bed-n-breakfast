const AWS = require("aws-sdk");
const dynamo = new AWS.DynamoDB({apiVersion: '2012-08-10'});
const s3 = new AWS.S3({ apiVersion: '2006-03-01' });

exports.handler = async (event) => {
  let requestJSON = JSON.parse(event.body);
  const base64File= requestJSON.roomImageFile;
     var BaseURL="https://group11roomimages.s3.amazonaws.com/";
     var ImageName="";
     if(base64File!="")
     {
        ImageName=`${new Date().toISOString()}.jpeg`
        const decodeFile= Buffer.from(base64File.replace(/^data:image\/\w+;base64,/,""),"base64");
        const S3params = {
                Bucket: "group11roomimages",
                Key:ImageName,
                Body: decodeFile,
                ContentType: 'image/jpeg',
                ACL:'public-read'
              };
        await s3.putObject(S3params).promise();
              var params = {
          TableName: 'RoomDetails',
          Item: {
            'id' :{S:event.queryStringParameters.id},
            'roomAmenity' :{S: requestJSON.roomAmenity},
            'noBeds' :{N:requestJSON.noBeds},
            'roomNo' :{S:requestJSON.roomNo},
            'roomPrice' :{N: requestJSON.roomPrice},
            'roomImageURL':{S: BaseURL + ImageName }
          }
        };
        var result = await dynamo.putItem(params).promise();
        result = JSON.stringify(result);
        return result;
     }
     else
     {
        var params = {
          TableName: 'RoomDetails',
          Item: {
            'id' :{S:event.queryStringParameters.id},
            'roomAmenity' :{S: requestJSON.roomAmenity},
            'noBeds' :{N:requestJSON.noBeds},
            'roomNo' :{S:requestJSON.roomNo},
            'roomPrice' :{N: requestJSON.roomPrice},
            'roomImageURL':{S: requestJSON.roomImageURL }
          }
        };
        var result = await dynamo.putItem(params).promise();
        result = JSON.stringify(result);
        return result;
     }
};
