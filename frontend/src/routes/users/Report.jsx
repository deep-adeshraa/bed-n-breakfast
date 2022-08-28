import React from "react";
import { useState, useEffect } from "react";
import axios from "axios";
import moment from "moment";

const Report = () => {
	const [user, setUser] = useState();
	const getData = async () => {
		const userlocal = JSON.parse(localStorage.getItem("user"));

		const response = await axios({
			method: "GET",
			url: `https://us-central1-serverless-project-356317.cloudfunctions.net/api/getUser?uid=${userlocal.userId}`,
		});
		setUser(response.data.user);
	};

	useEffect(() => {
		getData();
	}, []);

	return (
		<>
			<div className="container overflow-hidden">
				<div className="row mt-4">
					<div className="col-12 text-center">
						<h3>User Report Access Log</h3>
					</div>
				</div>
			</div>
			<div className="row mt-4 text-center">
				<div className="col-s12 text-center">
					<h4>User Email: {user && user.email}</h4>
				</div>
			</div>
			<div className="row mt-4 text-center">
				<div className="col-s6 text-center">
					<ul className="list-group">
						{user &&
							user.loginLog.map((log, index) => {
								return (
									<li className="list-group-item" key={index}>
										{moment(log).format("llll")}
									</li>
								);
							})}
					</ul>
				</div>
			</div>
		</>
	);
};

export default Report;
