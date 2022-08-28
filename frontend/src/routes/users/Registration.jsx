import React from "react";
import { useState, useEffect } from "react";
import { initializeApp } from "firebase/app";
import { getAuth, createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
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

const Registration = () => {
	const navigate = useNavigate();

	const [user, setUser] = useState({
		name: "",
		email: "",
		password: "",
		sec1: "",
		seca1: "",
		sec2: "",
		seca2: "",
	});

	const [loading, setLoading] = useState(false);
	const [registrationComplete, setRegistrationComplete] = useState(false);
	const [cipherKeyNumber, setCipherKeyNumber] = useState(-1);

	let userId;

	const registerUser = async (e) => {
		try {
			setLoading(true);
			// check for empty fields
			e.preventDefault();
			if (user.name === "" || user.email === "" || user.password === "" || user.sec1 === "" || user.seca1 === "" || user.sec2 === "" || user.seca2 === "") {
				alert("Please fill in all fields");
				setLoading(false);
				return;
			}

			// create firebase auth user
			const userCredential = await createUserWithEmailAndPassword(auth, user.email, user.password);
			await updateProfile(userCredential.user, {
				displayName: user.name,
			});
			userId = userCredential.user.uid;

			setTimeout(async () => {
				// set security questions
				const secResponse = await axios({
					method: "POST",
					url: "https://us-central1-serverless-project-356317.cloudfunctions.net/api/setSecurityQuestions",
					data: {
						uid: userId,
						securityQuestions: [
							{
								question: user.sec1,
								answer: user.seca1,
							},
							{
								question: user.sec2,
								answer: user.seca2,
							},
						],
					},
					headers: {
						"Content-Type": "application/json",
					},
				});
				setCipherKeyNumber(secResponse.data.user.cipherKeyNumber);
				setLoading(false);
				setRegistrationComplete(true);
			}, 3000);
		} catch (error) {
			console.log(error);
			setLoading(false);
			alert("Error registering user");
		}
	};

	return (
		<>
			<div className="container overflow-hidden">
				<div className="row mt-4 ">
					<div className="col-12 offset-5">
						<h3>Registration</h3>
					</div>
				</div>
				{!registrationComplete && (
					<>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="name">
										Full Name
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Enter Name Here"
										aria-label="Enter Name Here"
										aria-describedby="name"
										onChange={(e) => {
											setUser({ ...user, name: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="email">
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
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="sec1">
										Security Question 1
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Enter Question Here"
										aria-label="Enter Question Here"
										aria-describedby="sec1"
										onChange={(e) => {
											setUser({ ...user, sec1: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="seca1">
										Answer 1
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Enter Answer Here"
										aria-label="Enter Answer Here"
										aria-describedby="seca1"
										onChange={(e) => {
											setUser({ ...user, seca1: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="sec2">
										Security Question 2
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Enter Question Here"
										aria-label="Enter Question Here"
										aria-describedby="sec2"
										onChange={(e) => {
											setUser({ ...user, sec2: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-4">
							<div className="col-4 offset-4">
								<div className="input-group">
									<span className="input-group-text" id="seca2">
										Answer 2
									</span>
									<input
										type="text"
										className="form-control"
										placeholder="Enter Answer Here"
										aria-label="Enter Answer Here"
										aria-describedby="seca2"
										onChange={(e) => {
											setUser({ ...user, seca2: e.target.value });
										}}
									/>
								</div>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col-4 offset-4">
								{loading && (
									<div className="spinner-border" role="status">
										<span className="visually-hidden">Loading...</span>
									</div>
								)}

								{!loading && (
									<button type="button" className="btn btn-primary" onClick={registerUser}>
										Register
									</button>
								)}
							</div>
						</div>
					</>
				)}

				{registrationComplete && (
					<>
						<div className="row mt-5 text-success">
							<div className="col">
								<p>
									You are successfully registered!
									<br />
								</p>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col">
								<p>
									Your Cipher Key is <b>{cipherKeyNumber}</b>
									<br />
								</p>
							</div>
						</div>
						<div className="row mt-5">
							<div className="col">
								<button
									type="button"
									className="btn btn-primary"
									onClick={() => {
										navigate("/login");
									}}
								>
									Login
								</button>
							</div>
						</div>
					</>
				)}
			</div>
		</>
	);
};

export default Registration;
