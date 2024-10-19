Convolution Neural Networks

		1. Challenges with MLPs for Image Data
		
				● High dimensionality
				
				● Lack of spatial invariance
		
		2. Advantages of CNNs
		
				● Parameter sharing
				
				● Local connectivity3. Convolution Operation
				
				● Understanding Kernels/Filters
				
						○ Edge detection filters
						
						○ Feature extraction
		
				● Mathematical Representation
				
						○ Convolution in 2D and 3D
				
				● Hyperparameters
				
						○ Kernel size, depth
				
				● Stride and Padding
				
						○ Controlling output dimensions
						
						○ Types of padding: same vs. valid
		
		4. Activation Functions
		
				● ReLU (Rectified Linear Unit)
				
						○ Advantages over sigmoid/tanh
					
				● Variants
				
						○ Leaky ReLU
				
						○ ELU (Exponential Linear Unit)
		
		5. Pooling Layers
		
				● Purpose
				
						○ Dimensionality reduction
						
						○ Translation invariance
				
				● Types of Pooling
				
						○ Max pooling
						
						○ Average pooling
				
				● Pooling Size and Stride
		
		6. Fully Connected Layers
				
				● Transition from Convolutional Layers
				
				● Flattening
				
						○ Converting 2D features to 1D7. Loss Functions
				
				● Cross-Entropy Loss for Classification
				
				● Mean Squared Error for Regression
		
		8. CNN Architecture
			
			Layer Stacking
			
					● Convolutional -> Activation -> Pooling
			
			Feature Maps
			
					● Understanding depth and channels
			
			Visualization
			
					● Interpreting learned features
			
		9. Data Preprocessing Techniques - Data Normalization
				
				● Scaling Pixel Values
				
						○ 0-1 normalization
						
						○ Standardization (z-score)
		
		10. Data Preprocessing Techniques -Data Augmentation
		
				● Techniques
				
						○ Rotation, flipping, cropping
						
						○ Color jitter, noise addition
				
				● Purpose
				
						○ Reducing overfitting
						
						○ Increasing dataset diversity

CNN Architectures and Innovations

		1. LeNet-5
		
				● Architecture Details○ Layers, activations
				
				● Contributions
				
						○ Handwritten digit recognition
		
		2. AlexNet
		
				● Breakthroughs
				
						○ Deeper network
						
						○ Use of ReLU
				
				● Impact on ImageNet Challenge
		
		3. VGG Networks
		
				● VGG-16 and VGG-19
				
				● Design Philosophy
				
						○ Using small filters (3x3)
						
						○ Deep but uniform architecture
		
		4. Inception Networks (GoogLeNet)
		
				● Inception Modules
				
						○ Parallel convolutional layers
				
				● Motivation
				
						○ Efficient computation
		
		5. ResNet (Residual Networks)
		
				● Residual Blocks
				
						○ Identity mappings
						
						○ Shortcut connections
				
				● Solving Vanishing Gradient Problem
				
				● Variants
				
						○ ResNet-50, ResNet-101
		
		6. MobileNets
		
				● Depthwise Separable Convolutions● Optimizations for Mobile Devices
		
		7. Pre-trained Models & Transfer Learning
		
				● Using Models Trained on ImageNet
				
				● Fine-Tuning vs. Feature Extraction

Object Detection and Localization 

		1. Traditional Methods
		
				● Sliding Window Approach
		
		2. Modern Architecture
		
				● Region-Based CNNs (R-CNN)
				
						○ R-CNN
				
						○ Fast R-CNN
					
						○ Faster R-CNN
				
				● You Only Look Once (YOLO)
				
				● Single Shot MultiBox Detector (SSD)
				
				● Mask R-CNN
				
						○ Instance segmentation
				
			     ● Semantic Segmentation
		
		3. Fully Convolutional Networks (FCN)
		
				● Replacing Fully Connected Layers
		
		4. U-Net
		
				● Encoder-Decoder Architecture
				
				● Skip Connections

Generative Models with CNNs

		1. Autoencoders● Convolutional Autoencoders
		
				○ Image reconstruction
		
						● Variational Autoencoders (VAE)
		
		2. Generative Adversarial Networks (GANs)
				
				● DCGAN
				
						○ Using CNNs in GANs
				
				● Applications
				
						○ Image generation
						
						○ Super-resolution