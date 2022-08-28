import {useState,useEffect} from 'react'
import { useNavigate } from "react-router-dom";

const SearchRoom = () => {

    const navigate = useNavigate();
    const [toDate,settoDate]=useState('')
    const [noBeds,setnoBeds]=useState('')
    const [fromDate,setfromDate]=useState('')
    const [rooms,setrooms]=useState([])
    const [bookings,setbookings]=useState([])
    const [availableRooms,setavailableRooms]=useState([])
    const [state,setstate]=useState('none')
    const [bookingTotal,setbookingTotal]=useState(0)
    const [, updateState]=useState(); 
    const userData = JSON.parse(localStorage.getItem("user"));
    const sessionUserID =userData.userId;

    useEffect(() => {
      resetRooms();
      },[])

      const resetRooms = async () => {
        const RoomList = await getRoom()
        RoomList.forEach(function (room) {
            room.isAvailable = true;
        });
        setrooms(RoomList)
        const BookingList = await getBooking()
        setbookings(BookingList)
        setbookingTotal(BookingList.length);
        setstate('none');
        settoDate('');
        setnoBeds('');
        setfromDate('');
      }

      const getRoom = async () => {
        const res = await fetch('https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/getRoom',{"method": "GET"})
        const data = await res.json()  
        return data
      }  

      const getBooking = async () => {
        const res = await fetch('https://ni4d251mw1.execute-api.us-east-1.amazonaws.com/getRoomBooking',{"method": "GET"})
        const data = await res.json()  
        return data
      }
      
      const addRoomBooking = async (roomAdd) => {
        const res = await fetch('https://ni4d251mw1.execute-api.us-east-1.amazonaws.com/addRoomBooking',{"method": "POST",
        "body": JSON.stringify({
          "id" :roomAdd.id.toString(),
          "toDate" :roomAdd.toDate,
          "fromDate" :roomAdd.fromDate,
          "noBeds" :roomAdd.noBeds,
          "roomPrice" :roomAdd.roomPrice.toString(),
          "roomId":roomAdd.roomId,
          "customerId":roomAdd.customerId
        })})
        const data = await res.json()  
        return data
      } 

      const getTotalDays = (fromDate,toDate) => {
        var date1 = new Date(fromDate);
        var date2 = new Date(toDate);              
        var Difference_In_Time = date2.getTime() - date1.getTime();
        return (Difference_In_Time / (1000 * 3600 * 24));
      }

    const onClick = (e) =>{
        e.preventDefault();
        if(toDate!==""&&noBeds!==""&&fromDate!=="")
        {
            var i = rooms.length;
            while ( i --> 0 ) {
                if(rooms[i].noBeds >= noBeds)
                {
                    var c=availableRooms;
                    c.push(rooms[i]);
                    setavailableRooms(c); 
                }
            }

            var j = availableRooms.length;
            while ( j --> 0 ) {

                var roomID=availableRooms[j].id;
                var k = bookings.length;
                while ( k --> 0 ) {
                    if(bookings[k].roomId===roomID&&bookings[k].fromDate===fromDate&&bookings[k].toDate===toDate)
                    {
                        availableRooms[j].isAvailable=false;
                        break;
                    }
                    else if(bookings[k].roomId!==roomID&&bookings[k].fromDate!==fromDate&&bookings[k].toDate!==toDate)
                    {
                        availableRooms[j].isAvailable=true;
                    }
                }
            }
            setrooms(availableRooms);
            setstate('block');
            setavailableRooms([]);
            updateState({});
        }
        else
        {
            setstate('none');
            alert("Enter valid details")
        }      
    }

    const onBookClick =  (room) =>{
        return async function () {
        var totalRoomPrice=getTotalDays(fromDate,toDate) * room.roomPrice;
        debugger;
        await addRoomBooking({id:bookingTotal+10,toDate:toDate,fromDate:fromDate,noBeds:noBeds,roomPrice:totalRoomPrice,roomId:room.id,customerId:sessionUserID});
        alert("Room Booked Successfully");
        navigate("/my_reservation");
        resetRooms();
       }     
    }

  return (
    <div className="container mt-5">
    <div className="row">
      <div className="col-md-12">
        <h1 className="text-center" >Serverless Project: Group-11</h1>

        <div className="btn-group btn-group-lg w-100 mt-5" role="group" aria-label="Basic example">
          <a href="/kitchen" type="button" className="btn btn-secondary">Kitchen</a>
          <a href="/search_room" type="button" className="btn btn-secondary">Hotel</a>
          <a href="/book_tour" type="button" className="btn btn-secondary">Tour</a>
          <a href="/invoices" type="button" className="btn btn-secondary">My Bills</a>
          <a href="/my_orders" type="button" className="btn btn-secondary">My Orders</a>
          <a href="/my_reservation" type="button" className="btn btn-secondary">My Bookings</a>
        </div>
      </div>
    </div> <br/>
    <div className="row">
    <div className='col-md-6'>
    <div class="form-group">
        <label>From Date:</label>
        <input type="date" class="form-control" id="fromDate" placeholder="fromDate" value={fromDate} onChange={(e)=>setfromDate(e.target.value)}/>
    </div><br/>
    <div class="form-group">
        <label>To Date:</label>
        <input type="date" class="form-control" id="toDate" placeholder="toDate" value={toDate} onChange={(e)=>settoDate(e.target.value)}/>
    </div><br/>
    <div class="form-group">
        <label>No of Beds:</label>
        <input type="text" class="form-control" id="noBeds" placeholder="No of Beds" value={noBeds} onChange={(e)=>{
             const regularExpression = /^[+-]?([1-9]+\.?[0-9]*|\.[0-9]+)*$/
             if (regularExpression.test(e.target.value))
             {
                setnoBeds(e.target.value)
             }
            }}/>
    </div><br/>
        <input type='submit' onClick={onClick} className="btn btn-success" style={{"marginRight": "5px"}} value="Search"/>
        <button className='btn btn-primary' onClick={resetRooms}>Clear</button>
    </div></div>
    <br></br>

  { rooms.map((room)=>(
    <div class="card" style={{display: room.isAvailable ? "inline-block" : "none",width: 300 + "px",marginRight: 20 + "px",marginBottom: 20 + "px"}}>
    <img class="card-img-top" src={room.roomImageURL} alt="Card" style={{width: 300 + "px"}}/>
    <div class="card-body">
    <h4 class="card-title">${room.roomPrice}</h4>
    <p class="card-text">Room No: {room.roomNo}</p>
    <p class="card-text">Number of beds in room: {room.noBeds}</p>
    <p class="card-text">{room.roomAmenity}</p>
    <a class="btn btn-primary" style={{display:state}} onClick={onBookClick(room)}>Book</a>
    </div>
    </div>
            ))}
  </div>
  )
}

export default SearchRoom