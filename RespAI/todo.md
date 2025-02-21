# dataset
need to fix errors
- indexes -
	- 116
	- 127
	- 506
	- 554
	- 675
	- 691
	- 8564
- one third , | twice , thrice
- time - 24 hr
- day march 1 to April 19 - 37 days
- slash between numbers 2/3 
- 100 x 20/100 both 100 are replaced with 101
	

# influential unlearning





sorry sir we got busy and couldn't update sooner

for influential unlearning

Problems
- fine-tuned Llm's accuracy is poor.. can we use grokking method 
- Llm's responses are incomplete/ answer can't be extracted 
- in some of the responses the answer is son / daughter based on the person's gender ..but the dataset only has child as an answer.. should we convert son , daughter -> child

Todo
- need to prepare a graph for skeleton dataset.. then use agent to infer the indirect relationships between people
- top280 rank scatter plot
- derive a better method to match ground truth against the topk influential training points
- find auxiliary edges according to reasoning capabilities of the model

for symbolic dataset

update
- i scratched the old script and started again
- out of 8000  i got 7000 without any  major error
- also out of the 7000 i am getting small logic errors like the following that is messing up the data (lesser problems than the earlier dataset)
- currently  randomly sampling batch of points and checking and updating the script to accurately replace the datapoint without affecting other points

errors
- problems with days i,e  days between march 1 to April 19 is coming out  37 days 
- for some 100 x 20/100 both 100 are replaced with 101 ..here 20/100 is for 20 percentages
- sentences with one third ,  twice , thrice etc
- time - 24 hr
- errors with slash between numbers in the answer of datapoint ..2/3 or 3/4.. 
- etc