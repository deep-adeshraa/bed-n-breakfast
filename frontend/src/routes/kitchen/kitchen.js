import axios from "axios"
import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom";

function Kitchen(Props) {

  const [menu, setMenu] = useState([])
  const [order, setOrder] = useState([])
  const [images, setImages] = useState({});
  const navigate = useNavigate();
  const getMenu = async () => {
    const response = await axios.get('https://getmenunew-2ipzjcv5.ue.gateway.dev/get-breakfast-menu')
    setMenu(response.data.menu)
    getItemImage(response.data.menu)
  }

  useEffect(() => {
    getMenu();
  }, [])

  const addToOrder = (item) => {
    const hasItem = order.find(i => i.name === item.name)
    if (hasItem) {
      setOrder(order.map(i => {
        if (i.name === item.name) {
          i.quantity += 1
        }
        return i
      }
      ))
    }
    else {
      setOrder([...order, { ...item, quantity: 1 }])
    }
  }

  const remoteItemFromOrder = (item) => {
    setOrder(order.filter(i => i.name !== item.name))
  }

  const placeOrder = async () => {
    let user = localStorage.getItem("user");

    if (!user) {
      alert('Please login to place order');
      return;
    }
    user = JSON.parse(user);

    const response = await axios.post('https://addordersnew-2ipzjcv5.ue.gateway.dev/place-order', {
      orderItems: order,
      "orderPlaceBy": user.email,
      "firstName": user.displayName,
      "lastName": "",
      "total": order.reduce((acc, cur) => acc + cur.price * cur.quantity, 0)
    })
    alert(`Order placed successfully`)
    // clear order
    setOrder([])
    navigate("/my_orders")
  }

  const getItemImage = async (menu) => {
    let temp = {}
    console.log(menu)
    for (let i = 0; i < menu.length; i++) {
      const response = await axios.get(`https://4obu5002e0.execute-api.us-east-1.amazonaws.com/dev/getitemimg?id=${menu[i].id}`)
      temp[menu[i].id] = response.data.url;
    }
    console.log("temp"+temp)
    setImages(temp)
  }

  return (
    <div className="container">
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
      <h1 className="text-center mt-2">Kitchen</h1>
      <div className="row">
        {
          menu.map((item, index) => {
            return (
              <div className="col-md-3">
                <div className="card">
                  <img className="card-img-top" src={images[item.id]} style={{height:"200px"}} alt={item.name} />
                  <div className="card-body">
                    <h5 className="card-title">{item.name}</h5>
                    <p className="card-text">{item.price}$</p>
                    <button className="btn btn-primary" onClick={() => {
                      addToOrder(item)
                    }
                    }>Add to order</button>
                  </div>
                </div>
              </div>
            ) // end of return
          }
          ) // end of map
        }
      </div>
      <div>
        <h1 className="text-center mt-2">Order</h1>
        <div className="row">
          {
            order.map((item, index) => {
              return (
                <div className="col-md-3">
                  <div className="card">
                    <img className="card-img-top" src={images[item.id]} style={{height:"200px"}} alt={item.name} />
                    <div className="card-body">
                      <h5 className="card-title">{item.name}</h5>
                      <p className="card-text">Price : {item.price}$</p>
                      <p className="card-text">quantity: {item.quantity}</p>
                      <button className="btn btn-danger" onClick={() => {
                        remoteItemFromOrder(item)
                      }
                      }>Remove</button>
                    </div>
                  </div>
                </div>
              )
            }
            )
          }
        </div>
      </div>
      {order.length > 0 &&
        <div className="mt-5">
          <button className="btn btn-primary" onClick={(e) => placeOrder()} >Place Order</button>
        </div>}
    </div>
  )
}

export default Kitchen;
