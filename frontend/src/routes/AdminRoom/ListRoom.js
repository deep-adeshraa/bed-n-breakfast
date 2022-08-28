const ListRoom = (props) => {
  const {room,onEditroom,onDeleteroom} = props;

  const onEditClick = (room) =>{
    return function () {
      onEditroom(room);
   }     
}

const onDeleteClick = (ID) =>{
  return function () {
    onDeleteroom(ID);
 }     
}

  return (
    <tr>
      <td>{room.roomNo}</td>
      <td>{room.roomAmenity}</td>
      <td>{room.noBeds}</td>
      <td>{room.roomPrice}</td>
      <td><img src={room.roomImageURL} style={{width: 100 + "px"}}/></td>
      <td><input type='submit' className="btn btn-info" onClick={onEditClick(room)} style={{"marginRight": "5px"}} value='Edit'/><input type='submit' className="btn btn-danger" value='Delete' onClick={onDeleteClick(room.id)}/></td>
    </tr>
  )
}

export default ListRoom