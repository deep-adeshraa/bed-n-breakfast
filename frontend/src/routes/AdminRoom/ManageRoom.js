import AddRoom from "./AddRoom"
import {useState,useEffect}  from 'react'

const ManageRoom = () => {

  const [rooms,setrooms]=useState([])

  useEffect(() => {
    const getRooms = async () => {
    const RoomList = await getRoom()
    setrooms(RoomList)
  }
  getRooms()
  },[])

  const getRoom = async () => {
    const res = await fetch('https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/getRoom',{"method": "GET"})
    const data = await res.json()  
    return data
  }  
  
  const addRoom = async (roomAdd) => {
    const res = await fetch('https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/addRoom',{"method": "POST",
    "body": JSON.stringify({
      "id" :roomAdd.id.toString(),
      "roomAmenity" :roomAdd.roomAmenity,
      "noBeds" :roomAdd.noBeds,
      "roomNo" :roomAdd.roomNo,
      "roomPrice" :roomAdd.roomPrice,
      "roomImageFile":roomAdd.roomImageFile
    })})
    const data = await res.json()  
    return data
  } 
  
  const deleteRoom = async (ID) => {
    const res = await fetch(`https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/deleteRoom?id=${ID}`,{"method": "DELETE"})
    const data = await res.json()  
    return data
  } 
  
  const updateRoom = async (roomEdit) => {
    const res = await fetch(`https://pjwctivbz3.execute-api.us-east-1.amazonaws.com/updateRoom?id=${roomEdit.id}`,{"method": "PUT",
    "body": JSON.stringify({
      "roomAmenity" :roomEdit.roomAmenity,
      "noBeds" :roomEdit.noBeds.toString(),
      "roomNo" :roomEdit.roomNo,
      "roomPrice" :roomEdit.roomPrice.toString(),
      "roomImageFile":roomEdit.roomImageFile,
      "roomImageURL":roomEdit.roomImageURL
    })})
    const data = await res.json()  
    return data
  } 

  const AddroomCallback = async (roomAdd) =>{
    await addRoom(roomAdd);
    const RoomList = await getRoom()
    setrooms(RoomList)
}

const EditroomCallback = async (roomEdit) =>{ 
  await updateRoom(roomEdit);
  const RoomList = await getRoom()
  setrooms(RoomList)
}

const DeleteroomCallback = async (ID) =>{
  await deleteRoom(ID);
  const RoomList = await getRoom()
  setrooms(RoomList)
}

  return (
      <div>
    <AddRoom rooms={rooms} onAddroom={AddroomCallback} onEditroom={EditroomCallback} onDeleteroom={DeleteroomCallback} />
      </div>
  )
}

export default ManageRoom