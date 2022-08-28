function DataVisualization(props) {
  return (
    <div className="container mt-5">
        <div className="col-md-12">
        <h1 className="text-center">Data Visualization</h1>

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
     
      <hr />
      <div className="d-flex">
        <iframe className="mx-auto border rounded" width="1000" height="650" src="https://datastudio.google.com/embed/reporting/0e70c85b-f355-4676-8340-54e72d1c9110/page/tEnnC" frameborder="0" allowfullscreen></iframe>
      </div>
      <hr />
      <div className="d-flex">
        <iframe className="mx-auto border rounded" width="1000" height="650" src="https://datastudio.google.com/embed/reporting/0e4264a6-7a43-4f21-a13a-c11b8244808d/page/tEnnC" frameborder="0" allowfullscreen></iframe>
      </div>
      <hr />
      <div className="d-flex">
        <iframe className="mx-auto border rounded" width="1000" height="650" src="https://datastudio.google.com/embed/reporting/cd16cd51-3b5b-4279-bd1b-a466c9a33366/page/tEnnC" frameborder="0" allowfullscreen></iframe>
      </div>
    </div>
  )
}

export default DataVisualization;