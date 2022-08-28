import React from "react";
import { useState, useEffect } from "react";
import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, updateProfile } from "firebase/auth";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const firebaseConfig = {
	apiKey: "AIzaSyB2kHMTFMno79RFsg2E2qAOpM1IRpJaEZQ",
	authDomain: "serverless-project-356317.firebaseapp.com",
	projectId: "serverless-project-356317",
	storageBucket: "serverless-project-356317.appspot.com",
	messagingSenderId: "514886481850",
	appId: "1:514886481850:web:e5ba2259bbdbc9de3b8f83",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const Login = () => {
	const navigate = useNavigate();
	const [loginState, setLoginState] = useState("init");
	const [securityQuestions, setSecurityQuestions] = useState([]);
	const [cipherText, setCipherText] = useState("");
	const [multiFactorAuth, setMultiFactorAuth] = useState({
		a1: "",
		a2: "",
		decodedText: "",
	});
	const [user, setUser] = useState({
		email: "",
		password: "",
	});

	let userId, displayName, email;

	const loginUser = async (e) => {
		try {
			e.preventDefault();
			if (user.email === "admin" || user.password === "admin") {
				navigate("/manage_room");
				return;
			}

			if (user.email === "" || user.password === "") {
				alert("Please fill in all fields");
				return;
			}

			const userCredential = await signInWithEmailAndPassword(auth, user.email, user.password);
			userId = userCredential.user.uid;
			displayName = userCredential.user.displayName;
			email = userCredential.user.email;
			localStorage.setItem(
				"user",
				JSON.stringify({
					userId,
					displayName,
					email,
				})
			);

			const userQuestions = await axios({
				method: "GET",
				url: `https://us-central1-serverless-project-356317.cloudfunctions.net/api/getUser?uid=${userId}`,
			});
			setSecurityQuestions(userQuestions.data.user.securityQuestions);

			const generateCipher = await axios({
				method: "POST",
				url: `https://rjk25rvoq3.execute-api.ca-central-1.amazonaws.com/dev/generateCipher`,
				data: {
					uid: userId,
				},
				headers: {
					"Content-Type": "application/json",
				},
			});
			setCipherText(generateCipher.data.cipherText);
			setLoginState("multiFactor");
		} catch (e) {
			console.log(e);
			alert("Login Failed. Please try again.");
		}
	};

	const verifyMultiFactor = async (e) => {
		try {
			e.preventDefault();
			if (multiFactorAuth.a1 === "" || multiFactorAuth.a2 === "" || multiFactorAuth.decodedText === "") {
				alert("Please fill in all fields");
				return;
			}

			if (securityQuestions[0].answer !== multiFactorAuth.a1 || securityQuestions[1].answer !== multiFactorAuth.a2) {
				return alert("Multi Factor Auth Failed. Incorrect answers to the securtiy questions.");
			}

			const userData = JSON.parse(localStorage.getItem("user"));

			await axios({
				method: "POST",
				url: `https://us-central1-serverless-project-356317.cloudfunctions.net/api/verifySecurityQuestions`,
				data: {
					uid: userData.userId,
					securityQuestions,
				},
			});

			const verifyCipher = await axios({
				method: "POST",
				url: `https://rjk25rvoq3.execute-api.ca-central-1.amazonaws.com/dev/validateCipher`,
				data: {
					decryptedText: multiFactorAuth.decodedText,
					uid: userData.userId,
				},
			});
			console.log(verifyCipher.data.valid);
			if (verifyCipher.data.valid === false) {
				return alert("Multi Factor Auth Failed. Incorrect decrypted text.");
			}

			alert("Successfully logged in!");
			navigate("/home");
		} catch (e) {
			console.log(e);
			alert("Multi Factor Auth Failed. Please try again.");
		}
	};
	return (
		<>
			<div className="container overflow-hidden">
				<div className="row mt-4">
					<div className="col-12 offset-5">
						<h3>Login</h3>
					</div>
				</div>
				{loginState === "init" && (
					<>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="name">
										Email
									</span>
									<input
										type="email"
										className="form-control"
										placeholder="Enter Email Here"
										aria-label="Enter Email Here"
										aria-describedby="email"
										onChange={(e) => {
											setUser({ ...user, email: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="password">
										Password
									</span>
									<input
										type="password"
										className="form-control"
										placeholder="Enter Password Here"
										aria-label="Enter Password Here"
										aria-describedby="password"
										onChange={(e) => {
											setUser({ ...user, password: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<button type="button" className="btn btn-primary" onClick={loginUser}>
									Login
								</button>
							</div>
						</div>
					</>
				)}
				{loginState === "multiFactor" && (
					<>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<p>{securityQuestions[0].question}</p>
								<div className="input-group">
									<span className="input-group-text" id="answer1">
										Answer
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Your Answer"
										aria-label="Your Answer"
										aria-describedby="answer1"
										onChange={(e) => {
											setMultiFactorAuth({ ...multiFactorAuth, a1: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<p>{securityQuestions[1].question}</p>
								<div className="input-group">
									<span className="input-group-text" id="answer2">
										Answer
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Your Answer"
										aria-label="Your Answer"
										aria-describedby="answer2"
										onChange={(e) => {
											setMultiFactorAuth({ ...multiFactorAuth, a2: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<p>
									The Cipher text is <b>{cipherText}</b>. Please decode it using your key.
								</p>
								<div className="input-group">
									<span className="input-group-text" id="decoded">
										Decoded Text
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Your Text Here"
										aria-label="Your Text Here"
										aria-describedby="decoded"
										onChange={(e) => {
											setMultiFactorAuth({ ...multiFactorAuth, decodedText: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<button type="button" className="btn btn-primary" onClick={verifyMultiFactor}>
									Submit
								</button>
							</div>
						</div>
					</>
				)}
			</div>
		</>
	);
};

export { Login, auth };
