
decide if 
- running on device
	- quantize to 8 , 4 bit precision
	- find libraries to reduce inference time
	- add batch
- running on cloud
	- select the cloud service
	- decide 
		- if the model will be used as API
		- data will be sent to the model in batch and inferenced in batch
			- all the cropped images of face detected in one frame as one batch


### server - node

- [ ] deploy model
- [ ] add server client architecture
- [ ] send crop images as batch to -> deployed model
- [ ] add to database


### code
- [ ]  refactoring the code 	
- into class -
		- FaceDet 
		- FaceRecog
		- inbuilt function
		-
- [ ] attendance logic - padding   


### model
- [ ] use Deepface 
	- use one shot  
	- VGGface
- [ ] or use larger model


#### pytorch
- [ ] onnx
- [ ] quantize
- [ ] batch inference
- [ ] add bits and bytes for faster inference



### LLM
- [ ] Integrate LLM
- [ ] KV pressing for longer context for summarizing entire lecture
- [ ] quantize
- [ ] finetune
