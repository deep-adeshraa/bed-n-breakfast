const functions = require("firebase-functions");
const AWS = require("aws-sdk");
const cors = require("cors");
const express = require("express");

const API = express();

API.use(cors());

// set credentials for AWS
AWS.config.update({
	region: "ca-central-1",
	accessKeyId: functions.config().aws.accesskeyid,
	secretAccessKey: functions.config().aws.accesskey,
});

const dynamodb = new AWS.DynamoDB({
	region: "ca-central-1",
});

exports.registerUser = functions.auth.user().onCreate(async (user) => {
	// save user to dynamodb
	const params = {
		TableName: "serverless-project-users",
		Item: {
			uid: {
				S: user.uid,
			},
			email: {
				S: user.email,
			},
			cipherKeyNumber: {
				N: Math.floor(Math.random() * 10).toString(),
			},
			createdAt: {
				S: new Date().toISOString(),
			},
		},
	};
	return dynamodb.putItem(params).promise();
});

API.post("/setSecurityQuestions", async (req, res) => {
	try {
		// request body
		const uid = req.body.uid;
		const securityQuestions = req.body.securityQuestions;

		// update user security questions as array of objects
		const params = {
			TableName: "serverless-project-users",
			Key: {
				uid: {
					S: uid,
				},
			},
			UpdateExpression: "set securityQuestions = :securityQuestions",
			ExpressionAttributeValues: {
				":securityQuestions": AWS.DynamoDB.Converter.marshall({ securityQuestions }).securityQuestions,
			},
			ReturnValues: "ALL_NEW",
		};
		const user = await dynamodb.updateItem(params).promise();
		return res.json({
			success: true,
			message: "Security questions updated successfully",
			user: AWS.DynamoDB.Converter.unmarshall(user.Attributes),
		});
	} catch (e) {
		console.log(e);
		return res.status(500).send(e);
	}
});

API.post("/verifySecurityQuestions", async (req, res) => {
	const uid = req.body.uid;
	const securityQuestions = req.body.securityQuestions;

	// get user security questions
	const params = {
		TableName: "serverless-project-users",
		Key: {
			uid: {
				S: uid,
			},
		},
	};
	const user = await dynamodb.getItem(params).promise();
	const userSecurityQuestions = user.Item.securityQuestions.L;

	// compare user security questions with request body
	if (userSecurityQuestions.length !== securityQuestions.length) {
		return res.json({
			success: false,
			message: "Security questions do not match",
		});
	}
	for (let i = 0; i < userSecurityQuestions.length; i++) {
		if (userSecurityQuestions[i].M.question.S !== securityQuestions[i].question) {
			return res.json({
				success: false,
				message: "Security questions do not match",
			});
		}
		if (userSecurityQuestions[i].M.answer.S !== securityQuestions[i].answer) {
			return res.json({
				success: false,
				message: "Security questions do not match",
			});
		}
	}

	const docClient = new AWS.DynamoDB.DocumentClient({
		region: "ca-central-1",
	});

	const unUser = AWS.DynamoDB.Converter.unmarshall(user.Item);
	const loginLogs = unUser.loginLog ? unUser.loginLog : [];
	loginLogs.push(new Date().toISOString());

	// add user login log
	const params3 = {
		TableName: "serverless-project-users",
		Key: {
			uid,
		},
		UpdateExpression: "set loginLog = :loginLog",
		ExpressionAttributeValues: {
			":loginLog": loginLogs,
		},
		ReturnValues: "ALL_NEW",
	};
	await docClient.update(params3).promise();

	return res.json({
		success: true,
		message: "Security questions match",
	});
});

API.get("/getUser", async (req, res) => {
	const uid = req.query.uid;

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
	return res.json({
		success: true,
		user: AWS.DynamoDB.Converter.unmarshall(user.Item),
	});
});

API.get("/generateUsers", async (req, res) => {
	for (let i = 0; i < 10; i++) {
		const params = {
			TableName: "serverless-project-users",
			Item: {
				uid: {
					S: Math.floor(Math.random() * 100000).toString(),
				},
				email: {
					S: `${Math.floor(Math.random() * 100000).toString()}@gmail.com`,
				},
				cipherKeyNumber: {
					N: Math.floor(Math.random() * 100).toString(),
				},
				createdAt: {
					S: new Date(2022, 6, Math.floor(Math.random() * 30)).toISOString(),
				},
			},
		};
		await dynamodb.putItem(params).promise();
	}
	return res.json({
		success: true,
		message: "Users generated successfully",
	});
});

API.get("/loginUser", async (req, res) => {
	const userId = req.query.userId;

	// add user login log
	const params = {
		TableName: "serverless-project-users",
		Key: {
			uid: {
				S: userId,
			},
		},
		UpdateExpression: "set loginLog = list_append(loginLog, :loginLog)",
		ExpressionAttributeValues: {
			":loginLog": AWS.DynamoDB.Converter.marshall({
				loginAt: new Date().toISOString(),
			}),
		},
		ReturnValues: "ALL_NEW",
	};
	await dynamodb.updateItem(params).promise();
	return res.json({
		success: true,
	});
});

exports.api = functions.https.onRequest(API);
