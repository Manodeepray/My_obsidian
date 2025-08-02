![[Pasted image 20250706232620.png]]




- [ ]  need to read  
	- [ ] hyperLora
	- [ ] teleLora





just some thoughts on this algorithm ..unfortunately i have yet to properly read the hyperlora or telelora paper .. but i was looking into lora a little more and have some doubts i would like to share.. 

- in this algo I understand about increasing the retaining capabilities using lora..
- but the aim of TAR was to make it tamper resistant .. 
- it was observed that if someone's fine-tuning dataset overlaps with the harmful/forget dataset . it will ruin the model's performance which is good since we are trying to stop attackers
- but i have some doubt whether lora would do something like this
- as lora would only nudge the model ..
- as the base model may already have the harmful data or it will be trained/fine-tuned on the harmful data and used maliciously... 
- does adding the lora weights would have the same effect as the making it tamper resistant 
- because i think would definitely help retaining the benign knowledge , 

please correct me if i am mistaken.  