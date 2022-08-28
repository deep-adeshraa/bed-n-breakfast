import axios from "axios"
import { useEffect, useState } from "react"

function MyOrders(props) {
  const [orders, setOrders] = useState([]);

  const getOrders = async (email) => {
    const response = await axios.get(`https://get-order-2ipzjcv5.uc.gateway.dev/get-order?orderPlacedBy=${email}`);
    setOrders(response.data.orders);
    console.log(response.data.orders);
  }


  useEffect(() => {
    let user = localStorage.getItem("user");

    if (!user) {
      alert('Please login to view your orders');
      return;
    }
    user = JSON.parse(user);

    getOrders(user.email);
  }, []);


  return (
    <div>
   
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
      <h1 className="text-center">My Orders</h1>
      
      <div className="row">
        <div className="col-md-12">
          {
            orders.map((order, index) => {
              return (
                <div className="card mt-1">
                  <div className="card-body">
                    <h5 className="card-title">Order #{index + 1}</h5>
                    <p className="card-text">
                      <a style={{color:"green"}}><strong>Order Status:</strong> {order.orderCurrentStatus}</a><br />
                      
                      <strong>Order Placed By:</strong> {order.orderPlacedBy}<br />
                      <strong>Order Placed On:</strong> {order.orderPlacesTime.split('T')[0]+ " "+order.orderPlacesTime.split('T')[1].split(':')[0]+":"+order.orderPlacesTime.split(':')[1]}<br />
                      <strong>Order Ready On:</strong> {order.orderReadyTime.split('T')[0]+ " "+order.orderReadyTime.split('T')[1].split(':')[0]+":"+order.orderReadyTime.split(':')[1]}<br />
                      <strong>Order Total:</strong> {order.total}<br />
                      <strong>Order Items:</strong>
                      {
                        order.order.map((item, index) => {
                          return (
                            <div>
                              <p>{item.name} - {item.quantity}</p>
                            </div>
                          )
                        }
                        )
                      }
                    </p>
                  </div>
                </div>
              )
            })
          }
        </div>
      </div>
    </div>
   
    </div>
  )
}

export default MyOrders;
