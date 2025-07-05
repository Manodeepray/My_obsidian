










# Training Algorithm

Here is the extracted text from the image, structured in a clear, step-by-step format:

---

### **Optimization Process for TAR (Textual Adversarial Robustness)**  

1. **Initial Setup:**  
   - Start with a model parameterized by \( \theta \).  

2. **Parameter Copy:**  
   - Create a copy of the model’s parameters (\( \theta \)).  

3. **Adversarial Fine-Tuning (Inner Loop - \( k \) Steps):**  
   - Take the original model and perform \( k \) steps of adversarial fine-tuning:  
     - Use a batch of data to perform a forward pass.  
     - Compute gradients via backward pass.  
     - Update the model parameters to obtain \( \theta' \).  

4. **Proxy Safety Evaluation:**  
   - Sample a datapoint \( x_{TR} \) from a proxy dataset for safety evaluation.  
   - Input \( x_{TR} \) to the fine-tuned model \( \theta' \).  
   - Compute a loss (e.g., DPO loss or entropy loss \( \mathcal{L} \)) between the model’s output and ground truth.  

5. **Gradient Computation:**  
   - Compute gradients of \( \mathcal{L} \) w.r.t. the original parameters \( \theta \):  
     \[
     \nabla_{\theta} \mathcal{L}
     \]  

6. **Gradient Collection:**  
   - Store these gradients for later use.  

7. **Model Reversion:**  
   - Revert the model back to its original parameters \( \theta \).  

8. **Gradient Application:**  
   - Load the collected gradients into the \( \theta \).grad buffer.  

9. **Optimizer Step:**  
   - Update \( \theta \) by calling `optimizer.step()`.  

10. **Repeat:**  
    - Repeat steps 1–9 for \( j \) outer-loop iterations.  

---

### **Key Notes:**  
- **Objective:** Improve adversarial robustness while preserving safety metrics.  
- **Inner Loop:** Adversarial updates (\( k \) steps).  
- **Outer Loop:** Safety-aware meta-updates (\( j \) steps).  
- **Proxy Dataset:** Used to evaluate safety (e.g., harmful queries).  

This process balances adversarial training with safety constraints by iteratively fine-tuning and reverting the model.





# Algorithm from code from notebook

### images






