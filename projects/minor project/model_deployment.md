how can i deploy the model

if i send entire images 
 -> i can use the model as API 
	 image -> det model -> faces -> recog model -> attendees per frame -> 


if not using as api ..but as a continuous server-client connection
needs buffer : (images in buffer dir -> if buffer size full -> batch inference ->delete inferenced images feom buf dir -> cycle continues)
batch