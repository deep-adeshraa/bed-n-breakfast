import logo from "./logo.svg";
import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import ShowTours from "./routes/tour/show_tours";
import BookingConfirmation from "./routes/tour/booking_confirmation";
import Home from "./routes/home";
import Kitchen from "./routes/kitchen/kitchen";
import Invoices from "./routes/kitchen/invoices";
import MyOrders from "./routes/kitchen/my_orders";
import Registration from "./routes/users/Registration";
import { Login } from "./routes/users/Login";
import ManageRoom from "./routes/AdminRoom/ManageRoom";
import SearchRoom from "./routes/CustomerRoom/SearchRoom";
import MyTickets from "./routes/tour/MyTickets";
import Feedback from "./routes/FeedBack/FeedBack";
import UserFeedback from "./routes/FeedBack/UserFeedback";
import CustomerReservationList from "./routes/CustomerRoom/CustomerReservationList";
import DataVisualization from "./routes/visualization/data_visualization";
import Report from "./routes/users/Report";

function App() {
	return (
		<div className="App">
			<BrowserRouter>
				<Routes>
					<Route path="/home" element={<Home />} />
					<Route path="/book_tour" element={<BookingConfirmation />}></Route>
					<Route path="/show_tours" element={<ShowTours />}></Route>
					<Route path="/kitchen" element={<Kitchen />}></Route>
					<Route path="/invoices" element={<Invoices />}></Route>
					<Route path="/my_orders" element={<MyOrders />}></Route>
					<Route path="/login" element={<Login />}></Route>
					<Route path="/" element={<Registration />}></Route>
					<Route exact path="/manage_room" element={<ManageRoom />} />
					<Route
						path="/feedback"
						element={
							<div>
								<Home /> <Feedback />
							</div>
						}
					/>
					<Route
						path="/userfeedback"
						element={
							<div>
								 <UserFeedback />
							</div>
						}
					/>
					<Route exact path="/search_room" element={<SearchRoom />} />
					<Route exact path="/my_tickets" element={<MyTickets />}></Route>
					<Route exact path="/my_reservation" element={<CustomerReservationList />} />
					<Route exact path="/data_visualization" element={<DataVisualization />} />
					<Route exact path="/report" element={<Report />} />
				</Routes>
			</BrowserRouter>
		</div>
	);
}

export default App;
