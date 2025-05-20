# retrieval contact cards
- lit rev
- valulable info from each card
- prompt ten multi level rag -- if drive then automation - using smollm 
- universal rag + search
- box for image upload kyc example?
- from image -> text for card -> smart retrieval system 
- -> multimodal.. pics as well
- background search for each company and each person
	- linked in
	- other company webster --pyppeteer?
	- product
- autoscalable ec2 instance
	- kubernetes
- faster llm used inference features
	- locally run llm?
	- or API?
	- quantized
	- distilled smollm
- encyrypted rag
- byte latent rag
- cuda kernel for encoder?
- system designs interview book
- [[AI system design essentials in each project]]]
- encryption / security for each person
- database for each person
- compress image -> decompress for faster 
- low latency
- autocomplete sea
- screener


misrtal vlm ocr 
gemma 2b 
llava



# functionalities
1. add image into rag
2. search for text , images
3. have the search agent .. use MCP for searching the data on
4. one or more reasing loop 
5. chat history
6. fast inference on cloud
7. 

# workflow ->

1. image (askew)- properly angled + plus detected 
2. set up ocr models api on cloud --> stateless , 
3. setup ollama llm / groq api?
4. aws jwt auth for eact individual user
5. make database and add images --> simultaneously make  it user specific and divide it 
6. images to text
7. text to structured data via  llm 
8. structured data to vectorstore
9. create the multimodal vectorstore langgraph 
10. make the mm rag pipeline
11. set up evaluation pipeline
12. add bg search agent
13. run agent over already present cards ... add them to structured data then alll to vectorstore


14. search bar is important to write the name / --use llm to make a good prompt that woul allow the user to search more specifically
	1. e.g if name `ABC` is entered . .llm will make the leave this alone ...
	2. what if i dont remember the guy ...jut have a an idea what his name might be
	3. looking for company or details ike paint , steel worker --- llm will write the prompt such it can easy select them from company background , designation 
	4. etc



```
structured data = 
{
#image -> ocr model -> llmo  &&  image -> vlm -> card description
from_card:
	{
	name :,
	first_name:,
	sur_name:,
	middle_name:,
	contact info: 
		{
		emails:,
		ph nos:,
		},
	
	company:,
	designation:,
	card_details:   ,   #might use vlm to get this
	additonal info:,
	},

# agent will search these

individul_background:
	{
	linkedin:,
	google search:,
	other related links:,
	image:,
	},
	
company_background:
	{
	description: company website,
	reviews: yelp , justdial , sreener,
	linked in:,
	related links:,
	
	}
	
}
```


then go for other functionalities