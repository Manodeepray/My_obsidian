# DO UNLEARNING METHODS REMOVE INFORMATION FROM LANGUAGE MODEL WEIGHTS?

It has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access.

To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights

we give an attacker access to some facts that were supposed to be removed

the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts

fine-tuning on the accessible facts can ***recover 88% of the pre-unlearning accuracy*** when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights

![[Pasted image 20241103111609.png]]