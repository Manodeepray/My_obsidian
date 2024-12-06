#### Paper : [Understanding Black-Box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730)


to trace a model’s prediction through the learning algorithm and back to its training data, thereby identifying training points most responsible for a given prediction.

requires oracle access to gradients and Hessian-vector products.

even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information

use: linear models and convolutional neural networks, we demonstrate that influence functions are useful for multiple purposes: understanding model behavior, debugging models, detecting dataset errors, and even creating visually indistinguishable training-set attacks 


make the model explainable

Influence functions capture the core idea of studying models through the lens of their training data



Upweighting a training point:

goal : is to understand the effect of training points on a model’s predictions.
goal is formalized by asking the counterfactual : how would the models prediction change if we did not have this training point

retraining the model for each removed z is prohibitively slow


Relation to Euclidean distance :
To find the training points most relevant to a test point, it is common to look at its nearest neighbors in Euclidean space


problems to influence functions:
- One obstacle to adoption is that influence functions require expensive second derivative calculations and assume model differentiability and convexity, which limits their applicability in modern contexts where models are often non-differentiable, non-convex, and high dimensional.
- two challenges to efficiently computing Iup,loss(z, ztest) = −∇θL(ztest, ˆ θ) H−1 ˆ θ ∇θL(z, ˆ θ).
- First, it requires forming and inverting Hˆ θ = 1 n n i=1 ∇2 θL(zi, ˆ θ), the Hessian of the empirical risk. With n training points and θ ∈ Rp, this requires O(np2 + p3) operations, which is too expensive for models like deep neural networks with millions of parameters.
- Second, we need to calculate Iup,loss(zi, ztest) across all training points zi.

The idea is to avoid explicitly computing H−1 ˆ θ ; instead, we use implicit **Hessian-vector products** (HVPs) to efficiently approximate stest def = H−1 ˆ θ ∇θL(ztest, ˆ θ) and then compute Iup,loss(z,ztest) = −stest · ∇θL(z, ˆ θ). This also solves the second problem: for each test point of interest, we can precompute stest and then efficiently compute −stest · ∇θL(zi, ˆ θ) for each training point zi.


**Conjugate gradients (CG)**. The first technique is a standard transformation of matrix inversion into an optimization problem


**Stochastic estimation.** With large datasets, standard CG can be slow; each iteration still goes through all n training points. We use a method developed by Agarwal et al. (2016) to get an estimator that only samples a single point per iteration, which results in significant speedups.


### Use cases of influence functions

- Understanding model behavior
	- By telling us the training points “responsible” for a given prediction, influence functions reveal insights about how models rely on and extrapolate from the training data. In this section, we show that two models can make the same correct predictions but get there in very different ways.
- Adversarial training examples
	- we show that models that place a lot of inf luence on a small number of points can be vulnerable to training input perturbations, posing a serious security risk in real-world ML systems where attackers can influence the training data (Huang et al., 2011). Recent work has generated adversarial test images that are visually indistinguishable from real test images but completely fool a classifier (Goodfellow et al., 2015; Moosavi-Dezfooli et al., 2016). We demonstrate that influence functions can be used to craft adversarial training images that are similarly visuallyindistinguishable and can flip a model’s prediction on a separate test image. To the best of our knowledge, this is the f irst proof-of-concept that visually-indistinguishable training attacks can be executed on otherwise highly-accurate neural networks.
- Debugging domain mismatch
	- Domain mismatch — where the training distribution does not match the test distribution — can cause models with high training accuracy to do poorly on test data (Ben-David et al., 2010). We show that influence functions can identify the training examples most responsible for the errors, helping model developers identify domain mismatch.
- Fixing mislabeled examples
	- Labels in the real world are often noisy, especially if crowdsourced (Fr´ enay & Verleysen, 2014), and can even be adversarially corrupted, as in Section 5.2. Even if a human expert could recognize wrongly labeled examples, it is impossible in many applications to manually review all of the training data. We show that influence functions can help human experts prioritize their attention, allowing them to inspect only the examples that actually matter