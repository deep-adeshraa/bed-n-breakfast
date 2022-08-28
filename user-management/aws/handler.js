const AWS = require("aws-sdk");
const Response = require("./response");

const dynamodb = new AWS.DynamoDB({
	region: "ca-central-1",
});

module.exports.generateCipher = async (event, context) => {
	try {
		const { uid } = JSON.parse(event.body);

		// get user
		const params = {
			TableName: "serverless-project-users",
			Key: {
				uid: {
					S: uid,
				},
			},
		};
		const user = await dynamodb.getItem(params).promise();

		const cipherKeyNumber = parseInt(user.Item.cipherKeyNumber.N);

		// generate 5 character alphabetic random string
		let randomString = "";
		for (let i = 0; i < 5; i++) {
			randomString += String.fromCharCode(Math.floor(Math.random() * 26) + 97);
		}

		// encrypt using ceaser cipher
		let encryptedText = "";
		randomString.split("").forEach((char) => {
			const charCode = char.charCodeAt(0) - 96;
			const encryptedCharCode = (charCode + cipherKeyNumber) % 26 ? (charCode + cipherKeyNumber) % 26 : 26;
			const encryptedChar = String.fromCharCode(encryptedCharCode + 96);
			encryptedText += encryptedChar;
		});

		// save random string and encrypeted text to dynamodb append to the document
		const params2 = {
			TableName: "serverless-project-users",
			Key: {
				uid: {
					S: uid,
				},
			},
			UpdateExpression: "set encryptedText = :encryptedText, decryptedText = :decryptedText",
			ExpressionAttributeValues: {
				":encryptedText": {
					S: encryptedText,
				},
				":decryptedText": {
					S: randomString,
				},
			},
		};
		await dynamodb.updateItem(params2).promise();

		return Response.send(
			200,
			JSON.stringify({
				status: "success",
				cipherText: encryptedText,
			})
		);
	} catch (e) {
		console.log(e);
	}
};

module.exports.validateCipher = async (event, context) => {
	const { uid, decryptedText } = JSON.parse(event.body);

	// get user
	const params = {
		TableName: "serverless-project-users",
		Key: {
			uid: {
				S: uid,
			},
		},
	};
	const user = await dynamodb.getItem(params).promise();

	if (user.Item.decryptedText.S == decryptedText) {
		return Response.send(
			200,
			JSON.stringify({
				status: "success",
				valid: true,
			})
		);
	} else {
		return Response.send(
			200,
			JSON.stringify({
				status: "success",
				valid: false,
			})
		);
	}
};
