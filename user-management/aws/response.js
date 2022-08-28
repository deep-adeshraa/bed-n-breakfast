module.exports = {
	send(statusCode, body) {
		return {
			statusCode,
			headers: {
				"Content-Type": "application/json",
				"Access-Control-Allow-Origin": "*",
				"Access-Control-Allow-Credentials": true,
			},
			body,
		};
	},
};
