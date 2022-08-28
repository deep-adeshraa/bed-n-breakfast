import {useState} from 'react'
import ListRoom from './ListRoom';

const AddRoom = (props) => {

    const {rooms,onAddroom,onEditroom,onDeleteroom} = props;

    const [, updateState]=useState();

    const [id,setid]=useState(0)
    const [Status,setstatus]=useState('Add')
    const [roomAmenity,setroomAmenity]=useState('')
    const [noBeds,setnoBeds]=useState('')
    const [roomNo,setroomNo]=useState('')
    const [roomPrice,setroomPrice]=useState('')
    const [base64URL,setbase64URL]=useState('')
    const [roomImageURL,setroomImageURL]=useState('')

    const onClick = (e) =>{
        e.preventDefault()
        if(roomAmenity!==""&&noBeds!==""&&roomNo!==""&&roomPrice!=="")
        {
            if(id===0)
            {
                onAddroom({id:rooms.length+1,roomAmenity:roomAmenity,noBeds:noBeds,roomNo:roomNo,roomPrice:roomPrice,roomImageFile:base64URL});
            }
            else
            {
                onEditroom({id:id,roomAmenity:roomAmenity,noBeds:noBeds,roomNo:roomNo,roomPrice:roomPrice,roomImageFile:base64URL,roomImageURL:roomImageURL})               
            }
            updateState({});
            Clearroom();
        }
        else
        {
            alert("Enter valid details")
        }      
    }
    
    const EditroomCallback = (roomEdit) =>{
        setroomAmenity(roomEdit.roomAmenity);
        setnoBeds(roomEdit.noBeds);
        setroomNo(roomEdit.roomNo);
        setid(roomEdit.id);
        setroomPrice(roomEdit.roomPrice);
        setroomImageURL(roomEdit.roomImageURL);
        setstatus('Edit');
    }

    const DeleteroomCallback = (ID) =>{
        onDeleteroom(ID)
        updateState({});
        Clearroom();
    }

    const Clearroom = () =>{
        setroomAmenity('');
        setnoBeds('');
        setroomNo('');
        setid(0);
        setroomPrice('');
        setstatus('Add');
        setbase64URL('');
    }

  return (
    <div className="container mt-5">
    <div className="row">
      <div className="col-md-12">
        <h1 className="text-center" >Serverless Project: Group-11</h1>

        <div className="btn-group btn-group-lg w-100 mt-5" role="group" aria-label="Basic example">
          <a href="/manage_room" type="button" className="btn btn-secondary">Manage Rooms</a>
          <a href="/userfeedback" type="button" className="btn btn-secondary">
							See FeedBacks
						</a>
						<a href="/data_visualization" type="button" className="btn btn-secondary">
							See Data visualization
						</a>
          <a href="/login" type="button" className="btn btn-secondary">Logout</a>
        </div>
      </div>
    </div> <br/>
    <div className="row">
    <div className='col-md-6'>
    <div class="form-group">
        <label>Room No:</label>
        <input type="text" class="form-control" id="roomNo" placeholder="room No" value={roomNo} onChange={(e)=>{
             const regularExpression = /^[A-Za-z0-9]*$/;
             if (regularExpression.test(e.target.value))
             {
                setroomNo(e.target.value)
             }
            }}/>
    </div><br/>
    <div class="form-group">
        <label>Room Amenity:</label>
        <input type="text" class="form-control" id="roomAmenity" placeholder="room Amenity" value={roomAmenity} onChange={(e)=>{
             const regularExpression = /^[A-Za-z0-9 ]*$/;
             if (regularExpression.test(e.target.value))
             {
                setroomAmenity(e.target.value)
             }
            }}/>
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
    <div class="form-group">
        <label>Rent for 1 Day</label>
        <input type="text" class="form-control" id="roomPrice" placeholder="Price" value={roomPrice} onChange={(e)=>{
             const regularExpression = /^[+-]?([1-9]+\.?[0-9]*|\.[0-9]+)*$/
             if (regularExpression.test(e.target.value))
             {
                setroomPrice(e.target.value)
             }
            }}/>
    </div><br/>
    <div class="form-group">
        <label>Room Image</label>
        <input type="file" class="form-control-file" id="roomImage" accept="image/*" onChange={(e)=>{         
            let reader = new FileReader();
            reader.readAsDataURL(e.target.files[0]);
            reader.onload =  function () {
                 setbase64URL(reader.result)
            };   
            }}/>
    </div><br/>
        <input type='submit' onClick={onClick} className="btn btn-success" style={{"marginRight": "5px"}} value={Status}/><button className='btn btn-primary' onClick={Clearroom}>Clear</button>
    </div></div>

    <br/>
    <table class="table table-bordered">
    <thead>
      <tr>
        <th>Room No</th>
        <th>Room Amenity</th>
        <th>No of Beds</th>
        <th>Rent for 1 Day</th>
        <th>Image</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>
            { rooms.length > 0 ? rooms.map((room)=>(
            <ListRoom key={room.id} room={room} onEditroom={EditroomCallback} onDeleteroom={DeleteroomCallback}/>
            )): <tr><td colSpan="4">No rooms</td></tr>}
    </tbody>
   </table>
        </div>
  )
}

export default AddRoom