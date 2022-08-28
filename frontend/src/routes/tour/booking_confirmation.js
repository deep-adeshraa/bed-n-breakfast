import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import moment from 'moment';
import axios from "axios";

function BookingConfirmation(props) {
  const [startDate, setStartDate] = useState();
  const [endDate, setEndDate] = useState();
  const [tourPlace, setTourPlace] = useState('Halifax');
  const navigate = useNavigate();

  const confirmBooking = () => {

    if (!startDate || !endDate) {
      return;
    }

    var start = moment(startDate, "YYYY-MM-DD");
    var end = moment(endDate, "YYYY-MM-DD");

    //Difference in number of days
    var days = moment.duration(end.diff(start)).asDays();

    var data = {
      'place': tourPlace,
      'days': days,
      'start_date': startDate,
      'end_date': endDate
    }

    navigate('/show_tours', { state: data })
  }

  return (
    <div className="container text-center">
      <h1 className="mt-3">Select place and dates</h1>
      <div className="card mt-4 col-md-8 mx-auto">
        {/* <img className="card-img-top mt-2" src={item.image} alt={item.place} /> */}
        <div className="card-body">
          {/* <h5 className="card-title">{item.place}</h5>
          <p className="card-text">{item.description}</p>
          <h5 className="card-title">Price : {tourPrice}$</h5> */}

          <strong className="pull-left mt-3"><label className="pull-left mt-3">Start date</label></strong>
          <input onChange={(e) => setStartDate(e.target.value)} className="form-control" type="date"></input>

          <strong className="pull-left mt-3"><label className="mt-3">End date</label></strong>
          <input onChange={(e) => setEndDate(e.target.value)} min={startDate} className="form-control" type="date"></input>

          <strong className="pull-left mt-3"><label className="mt-3">Place</label></strong>
          <select className="form-control" onChange={(e) => setTourPlace(e.target.value)}>
            <option className="form-option">Halifax</option>
            <option className="form-option">Hawai</option>
            <option className="form-option">San fansisco</option>
            <option className="form-option">New Yourk</option>
            <option className="form-option">Delhi</option>
            <option className="form-option">Mumbai</option>
            <option className="form-option">Ghana</option>
            <option className="form-option">Shilanka</option>
            <option className="form-option">Thailand</option>
            <option className="form-option">Hong Kong</option>
            <option className="form-option">Syndney</option>
            <option className="form-option">Hemilton</option>
            <option className="form-option">Vancouver</option>
            <option className="form-option">Seattle</option>
            <option className="form-option">Ottawa</option>
            <option className="form-option">Jamaica</option>
            <option className="form-option">Lebenon</option>
          </select>
          {
            startDate && endDate &&
            <a onClick={confirmBooking} className="btn btn-primary mt-4">Confirm</a>
          }
        </div>
      </div>
    </div>
  )
}

export default BookingConfirmation;