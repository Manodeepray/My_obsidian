- **Stop Re-identifying Canaries:** Your immediate priority is to restructure your training loop to perform canary identification only once, before the main harmless training begins.
    
- **Separate Adversarial and Harmless Training:** The adversarial phase is for _finding_ the safety configuration. The harmless phase is for _general training while preserving_ that configuration. Do not mix them inside the same epoch loop.
    
- **Re-evaluate Loss After Fixing the Loop:** Once the loop is corrected, your `Harmless Loss` will likely become much more stable and start to decrease properly. If it's still too high, you can then tune hyperparameters like the `STABILIZATION_LAMBDA` and the learning rate.

- [ ] fix

