import axios from 'axios';
import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function largestIndex(array) {
  var counter = 1;
  var max = 0;

  for (counter; counter < array.length; counter++) {
    if (array[max] < array[counter]) {
      max = counter;
    }
  }
  return max;
}

function ShowTours(props) {
  const [availablePlaces, setAvailablePlaces] = useState([]);
  const location = useLocation();
  const [data, setData] = useState({});
  const [prediction, setPrediction] = useState();
  const [price, setPrice] = useState();

  const getData = async (data) => {
    setData(data)
    const response = await axios.get('https://us-central1-peak-service-312506.cloudfunctions.net/get_available_places');
    setAvailablePlaces(response.data.data);
    predictData(data.place, data.days)
  }

  const predictData = async (place, days) => {
    const response = await axios.get(`https://us-central1-peak-service-312506.cloudfunctions.net/predict-prices?place=${place}&days=${days}`)
    setPrediction(response.data.data);
    const index = largestIndex(response.data.data.scores)
    setPrice(response.data.data.classes[index])
  }

  const bookTicket = async (item) => {
    let user = localStorage.getItem("user");

    if (!user) {
      alert('Please login to book a ticket')
      return;
    }
    user = JSON.parse(user);

    const response = await axios.get('https://us-central1-peak-service-312506.cloudfunctions.net/create-ticket', {
      params: {
        place: item.place,
        start_date: data.start_date,
        end_date: data.end_date,
        price: price,
        user_id: user.email
      }
    })

    if (response.data.success) {
      alert("Ticket booked successfully")
    } else {
      alert(response.data.message)
    }
  }

  useEffect(() => {
    getData(location.state);
    console.log(location)
  }, [])

  return (
    <div className="container mt-5">
      <h1 className='mb-3'>Available tour packages</h1>
      <div className="col-md-12 row mb-4">
        {
          availablePlaces.map((item, index) => {
            if (item.place.toLowerCase() != data?.place?.toLowerCase()) {
              return
            }

            return (
              <div className="card col-md-6 mx-auto">
                <img className="card-img-top mt-2" src={item.image} alt={item.place} />
                <div className="card-body">
                  <h5 className="card-title">{item.place}</h5>
                  <strong><p className="card-text">Predicted Price: {price}$</p></strong>
                  <p className="card-text">{item.description}</p>
                </div>

                <div className="card-footer">
                  <a href="#" onClick={() => bookTicket(item)} className="btn btn-primary">Book Now</a>
                </div>
              </div>
            );
          })
        }
      </div>
    </div>
  )
}

export default ShowTours;