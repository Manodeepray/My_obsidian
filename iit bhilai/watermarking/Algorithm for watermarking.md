

Algorithm for watermarking:
1. Take RGB MRI image with detected Tumor .The ROI shoud be a bounding Box

![[Pasted image 20250222010451.png]]

2. Separate the channels into a  tensor of shape -  
	 RGB_img.shape() = [ x , y, 3 ]
	 img :  width = x , height = y

![[Pasted image 20250222010546.png]]

3. Store the coordinate of the bounding box as  Roi_coords = [Lu , Lb , Ru , Rb] 
4. Create a Zero matrix of shape ( x, y, ) , which will be added as watermark channel 
	 wm.shape() = [x, y] 
5. Fill the coordinate of ROI in the new watermark with a certain pixel value like (1 or 255)

![[Pasted image 20250222011016.png]]



6. Choose a method of data encoding for dynamic watermarking
7. encode the meta-data , make sure the encoded data is not the same as roi pixel values of the watermark
8. store the row wise encoded data in the new watermark channel 
	1. row-wise : L->R or R->L : starting from top row to bottom row or vice versa 
	2. column-wise : T->B or B->T :  starting from Leftmost column to rightmost column or vice versa
![[Pasted image 20250222011111.png]]
9. Add this Watermark to the image tensor 
	 watermarked_image.shape() = [ x, y, 4 ]
![[Pasted image 20250222011154.png]]
10. find a way to decode