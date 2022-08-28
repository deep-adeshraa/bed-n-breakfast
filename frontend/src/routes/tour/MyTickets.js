import axios from "axios";
import { useEffect, useState } from "react";


function MyTickets(props) {

  const [tickets, setTickets] = useState([]);

  const getTickts = async (user) => {
    const response = await axios.get(`https://us-central1-peak-service-312506.cloudfunctions.net/get_my_tickets?user_id=${user}`);
    setTickets(response.data.tickets);
  }

  useEffect(() => {
    let user = localStorage.getItem("user");

    if (!user) {
      alert('Please login to view your tickets');
      return;
    }

    user = JSON.parse(user);
    getTickts(user.email);
  }, []);


  return (
    <div>
      <h1 className="text-center mt-4">My Tickets</h1>
      <div className="container mt-5">
        <div className="row">
          <div className="col-md-12">
            <table className="table table-striped">
              <thead>
                <tr>
                  <th>Place</th>
                  <th>Price</th>
                  <th>start date</th>
                  <th>end date</th>
                </tr>
              </thead>
              <tbody>
                {tickets.map(ticket => (
                  <tr key={ticket.id}>
                    <td>{ticket.place}</td>
                    <td>{ticket.price}$</td>
                    <td>{ticket.start_date}</td>
                    <td>{ticket.end_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MyTickets;