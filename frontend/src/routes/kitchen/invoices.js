import axios from "axios";
import { useEffect, useState } from "react";

function Invoices(props) {

  const [invoices, setInvoices] = useState([]);

  const getInvoices = async (email) => {
    const response = await axios.get(`https://k15skxig3b.execute-api.us-east-1.amazonaws.com/dev/getinvoices?id=${email}`);
    setInvoices(response.data.invoices);
    console.log(response.data.invoices)
  }

  useEffect(() => {
    let user = localStorage.getItem("user");

    if (!user) {
      alert('Please login to view your tickets');
      return;
    }

    user = JSON.parse(user);
    getInvoices(user.email);
  }, [])

  return (
    <div className="container mt-5">
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
      <h1 className="text-center">My Bills</h1>
      <div className="row">
        <div className="col-md-12">
          {
            invoices.map((invoice, index) => {
              return (<div className="card mt-1">
                <div className="card-body">
                  <h5 className="card-title">Invoice #{index + 1}</h5>
                  <a href={invoice} target="_blank" className="btn btn-primary">View Invoice</a>
                </div>
              </div>)
            })
          }
        </div>
      </div>
    </div>

  )
}

export default Invoices;
