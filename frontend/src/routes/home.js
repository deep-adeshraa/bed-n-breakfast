function Home(props) {
	return (
		<div className="container mt-5">
			{/* Add button group  */}
			<div className="row">
				<div className="col-md-12">
					<h1 className="text-center">Serverless Project: Group-11</h1>

					<div className="btn-group btn-group-lg w-100 mt-5" role="group" aria-label="Basic example">
						<a href="/kitchen" type="button" className="btn btn-secondary">
							Kitchen
						</a>
						<a href="/search_room" type="button" className="btn btn-secondary">
							Hotel
						</a>
						<a href="/book_tour" type="button" className="btn btn-secondary">
							Tour
						</a>
						<a href="/invoices" type="button" className="btn btn-secondary">
							My Bills
						</a>
						<a href="/my_orders" type="button" className="btn btn-secondary">
							My Orders
						</a>
					</div>
					<div className="btn-group btn-group-lg w-100 mt-5" role="group" aria-label="Basic example">
						<a href="/my_reservation" type="button" className="btn btn-secondary">
							My Room Bookings
						</a>
						<a href="/feedback" type="button" className="btn btn-secondary">
							Write FeedBack
						</a>
						<a href="/my_tickets" type="button" className="btn btn-secondary">
							My tour bookings
						</a>
						
						<a href="/report" type="button" className="btn btn-secondary">
							Access Report
						</a>
					</div>
				</div>
			</div>
		</div>
	);
}

export default Home;
