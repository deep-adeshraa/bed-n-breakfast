import { useState, useEffect } from 'react'

const UserFeedback = () => {


  const [feedback, setFeedback] = useState([]);
  useEffect(() => {
   
    const getFeedback = async () => {
      const Feedbacklist = await getFeedbacks()
      setFeedback(Feedbacklist);
    }
    getFeedback()
  }, [])

 
  const getFeedbacks = async () => {
    const res = await fetch('https://4ucj1xcie9.execute-api.us-east-1.amazonaws.com/feedback', { "method": "GET" })
    const data = await res.json()
    //console.log(res);
    return data

  }



  return (
    <div  className="container mt-4">
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
      <br/>
        <table className="table table-bordered table-hover" >
          <thead class="table-success">
            <tr>
              <th>User Name</th>
              <th>Feedback</th>
              <th>Sentiment</th>
            </tr>
          </thead>
          <tbody>
            {feedback.length > 0 ? feedback.map((feedback) => (

              <tr>
                <td>{feedback.name}</td>
                <td>{feedback.feedback}</td>
                <td>{feedback.sentiment}</td>
              </tr>
            )) : <tr>
              <td colSpan="4">No Feedbacks</td>
            </tr>}
          </tbody>
        </table>
      
    </div>
  )
}

export default UserFeedback;
