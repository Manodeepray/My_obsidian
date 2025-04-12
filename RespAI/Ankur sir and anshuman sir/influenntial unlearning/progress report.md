

# influential unlearning
## done

1.deep unlearning dataset adopted for datainf script
	for finetuning
		 training dataset
		 testing dataset	
2.finetuned llm -llama 13b in the train dataset

3.evaluated how accurately the finetuned llama model answer the relatiohship questions against the base model (untouched)

4.calculated the influence 
	of each training dataset data point for each testing dataset datapoint

5,sorted the training point 
	top 10
	 top 20
	 top 50
	 top 280(entire training dataset)
	(leave space for images)

6.created a graph showing all the relationship between each member i.e prepared a graph for skeleton dataset
(leave space for images)

## problems faced


1.- fine-tuned Llm's accuracy is poor.. (l;eave space for accuracies)

2.- Llm's responses are incomplete/ answer can't be extracted (leave space for images)
- 
3.- in some of the responses the answer is son / daughter based on the person's gender ..but the dataset only has child as an answer.. should we convert son , daughter -> child

### Todo

1.- finetune newre model i.e qwen or deepseek
2.- use agent to infer the indirect relationships between people from the skeleton dataset
3.- top280 rank scatter plot
4.- derive a better method to match ground truth against the topk influential training 5.points
6.- find auxiliary edges according to reasoning capabilities of the model
- 



# symbolic dataset

## Done

 i scratched the old script and started again
- out of 8000  i got around 6000 without any  major error
- also out of the 6000 i am getting small logic errors like the following that is messing up the data (fewer problems than the previous dataset)
- currently  randomly sampling batch of points and checking and updating the script to accurately replace the datapoint without affecting other points
## problems faced

- problems with parsing logic regarding days i,e  days between march 1 to April 19 is coming out  37 days after replacing
- for some percentage..in the answer we have 100 x 20/100..here both 100 are replaced with 101 intead on only the 100 in the numerator
- parsing sentences with one third ,  twice , thrice etc
- problems time in  24 hr
- couple of errors with forward slash between numbers in the answer ..2/3 or 3/4..




