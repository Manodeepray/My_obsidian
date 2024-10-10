
**Problem framing** is the process of analyzing a problem to isolate the individual elements that need to be addressed to solve it. Problem framing helps determine your project's technical feasibility and provides a clear set of goals and success criteria. When considering an ML solution, effective problem framing can determine whether or not your product ultimately succeeds.


> Formal problem framing is the critical beginning for solving an ML problem, as it forces us to better understand both the problem and the data in order to design and build a bridge between them. - _TensorFlow engineer_

At a high level, ML problem framing consists of two distinct steps:

1. Determining whether ML is the right approach for solving a problem.
2. Framing the problem in ML terms.


## Understand the problem

To understand the problem, perform the following tasks:

- State the goal for the product you are developing or refactoring.
- Determine whether the goal is best solved using, predictive ML, generative AI, or a non-ML solution.
- Verify you have the data required to train a model if you're using a predictive ML approach.
## State the goal

Begin by stating your goal in non-ML terms. The goal is the answer to the question, "What am I trying to accomplish?"

The following table clearly states goals for hypothetical apps:

| **Application** | **Goal**                                                                |
| --------------- | ----------------------------------------------------------------------- |
| Weather app     | Calculate precipitation in six-hour increments for a geographic region. |
| Fashion app     | Generate a variety of shirt designs.                                    |
| Video app       | Recommend useful videos.                                                |
| Mail app        | Detect spam.                                                            |
| Financial app   | Summarize financial information from multiple news sources.             |
| Map app         | Calculate travel time.                                                  |
| Banking app     | Identify fraudulent transactions.                                       |
| Dining app      | Identify cuisine by a restaurant's menu.                                |
| Ecommerce app   | Reply to reviews with helpful answers.                                  |


## Clear use case for ML

Some view ML as a universal tool that can be applied to all problems. In reality, ML is a specialized tool suitable only for particular problems. You don't want to implement a complex ML solution when a simpler non-ML solution will work.



ML systems can be divided into two broad categories: [**predictive ML**](https://developers.google.com/machine-learning/glossary#predictive-ml) and [**generative AI**](https://developers.google.com/machine-learning/glossary#generative-ai). The following table lists their defining characteristics:


|                   | **Input**                                            | **Output**                                                                                                                                                                                         | **Training technique**                                                                                                                                                                                                                                                                                    |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Predictive ML** | Text  <br>Image  <br>Audio  <br>Video  <br>Numerical | Makes a prediction, for example, classifying an email as spam or not spam, guessing tomorrow's rainfall, or predicting the price of a stock. The output can typically be verified against reality. | Typically uses lots of data to train a supervised, unsupervised, or reinforcement learning model to perform a specific task.                                                                                                                                                                              |
| **Generative AI** | Text  <br>Image  <br>Audio  <br>Video  <br>Numerical | Generates output based on the user's intent, for example, summarizing an article, or producing an audio clip or short video.                                                                       | Typically uses lots of unlabeled data to train a large language model or image generator to fill in missing data. The model can then be used for tasks that can be framed as fill-in-the-blank tasks, or it can be fine-tuned by training it on labeled data for some specific task, like classification. |
To confirm that ML is the right approach, first verify that your current non-ML solution is optimized. If you don't have a non-ML solution implemented, try solving the problem manually using a [**heuristic**](https://developers.google.com/machine-learning/glossary#heuristic).

## heuristic

A simple and quickly implemented solution to a problem. For example, "With a heuristic, we achieved 86% accuracy. When we switched to a deep neural network, accuracy went up to 98%."

The non-ML solution is the benchmark you'll use to determine whether ML is a good use case for your problem.

Consider the following questions when comparing a non-ML approach to an ML one:

- **Quality**. How much better do you think an ML solution can be? If you think an ML solution might be only a small improvement, that might indicate the current solution is the best one.
    
- **Cost and maintenance**. How expensive is the ML solution in both the short- and long-term? In some cases, it costs significantly more in terms of compute resources and time to implement ML. Consider the following questions:
    
    - Can the ML solution justify the increase in cost? Note that small improvements in large systems can easily justify the cost and maintenance of implementing an ML solution.
    - How much maintenance will the solution require? In many cases, ML implementations need dedicated long-term maintenance.
    - Does your product have the resources to support training or hiring people with ML expertise?


## Predictive ML and data

Data is the driving force of predictive ML. To make good [**predictions**](https://developers.google.com/machine-learning/glossary#prediction), you need data that contains [**features**](https://developers.google.com/machine-learning/glossary#feature) with predictive power. Your data should have the following characteristics:

- **Abundant**. The more relevant and useful examples in your [**dataset**](https://developers.google.com/machine-learning/glossary#data-set-or-dataset), the better your model will be.
    
- **Consistent and reliable**. Having data that's consistently and reliably collected will produce a better model. For example, an ML-based weather model will benefit from data gathered over many years from the same reliable instruments.
    
- **Trusted**. Understand where your data will come from. Will the data be from trusted sources you control, like logs from your product, or will it be from sources you don't have much insight into, like the output from another ML system?
    
- **Available**. Make sure all inputs are available at prediction time in the correct format. If it will be difficult to obtain certain feature values at prediction time, omit those features from your datasets.
    
- **Correct**. In large datasets, it's inevitable that some [**labels**](https://developers.google.com/machine-learning/glossary#label) will have incorrect values, but if more than a small percentage of labels are incorrect, the model will produce poor predictions.
    
- **Representative**. The datasets should be as representative of the real world as possible. In other words, the datasets should accurately reflect the events, user behaviors, and/or the phenomena of the real world being modeled. Training on unrepresentative datasets can cause poor performance when the model is asked to make real-world predictions.
    

If you can't get the data you need in the required format, your model will make poor predictions.



### Predictive power

**For a model to make good predictions, the features in your dataset should have predictive power. The more correlated a feature is with a label, the more likely it is to predict it.**

Some features will have more predictive power than others. For example, in a weather dataset, features such as `cloud_coverage`, `temperature`, and `dew_point` would be better predictors of rain than `moon_phase` or `day_of_week`. For the video app example, you could hypothesize that features such as `video_description`, `length` and `views` might be good predictors for which videos a user would want to watch.

Be aware that a feature's predictive power can change because the context or domain changes. For example, in the video app, a feature like `upload_date` might—in general—be weakly correlated with the label. However, in the sub-domain of gaming videos, `upload_date` might be strongly correlated with the label.

Determining which features have predictive power can be a time consuming process. You can manually explore a feature's predictive power by removing and adding it while training a model. You can automate finding a feature's predictive power by using algorithms such as [[pearson's correlation]], [[Adjusted mutual information]], and [[Shapley value]], which provide a numerical assessment for analyzing the predictive power of a feature.

## Framing an ML problem
After verifying that your problem is best solved using either a predictive ML or a generative AI approach, you're ready to frame your problem in ML terms. You frame a problem in ML terms by completing the following tasks:

- Define the ideal outcome and the model's goal.
- Identify the model's output.
- Define success metrics.

## Define the ideal outcome and the model's goal

Independent of the ML model, what's the ideal outcome? In other words, what is the exact task you want your product or feature to perform? This is the same statement you previously defined in the [State the goal](https://developers.google.com/machine-learning/problem-framing/problem#state_the_goal) section.

Connect the model's goal to the ideal outcome by explicitly defining what you want the model to do. The following table states the ideal outcomes and the model's goal for hypothetical apps:



|**App**|**Ideal outcome**|**Model's goal**|
|---|---|---|
|Weather app|Calculate precipitation in six hour increments for a geographic region.|Predict six-hour precipitation amounts for specific geographic regions.|
|Fashion app|Generate a variety of shirt designs.|Generate three varieties of a shirt design from text and an image, where the text states the style and color and the image is the type of shirt (t-shirt, button-up, polo).|
|Video app|Recommend useful videos.|Predict whether a user will click on a video.|
|Mail app|Detect spam.|Predict whether or not an email is spam.|
|Financial app|Summarize financial information from multiple news sources.|Generate 50-word summaries of the major financial trends from the previous seven days.|
|Map app|Calculate travel time.|Predict how long it will take to travel between two points.|
|Banking app|Identify fraudulent transactions.|Predict if a transaction was made by the card holder.|
|Dining app|Identify cuisine by a restaurant's menu.|Predict the type of restaurant.|
|Ecommerce app|Generate customer support replies about the company's products.|Generate replies using sentiment analysis and the organization's knowledge base.|

### Identify the output you need

Your choice of model type depends upon the specific context and constraints of your problem. The model's output should accomplish the task defined in the ideal outcome. Thus, the first question to answer is "What type of output do I need to solve my problem?"

If you need to classify something or make a numeric prediction, you'll probably use predictive ML. If you need to generate new content or produce output related to natural language understanding, you'll probably use generative AI.

The following tables list predictive ML and generative AI outputs:

|   |   |   |
|---|---|---|
**Table 1.** Predictive ML
||ML system|Example output|
|Classification|Binary|Classify an email as spam or not spam.|
|Multiclass single-label|Classify an animal in an image.|
|Multiclass multi-label|Classify all the animals in an image.|
|Numerical|Unidimensional regression|Predict the number of views a video will get.|
|Multidimensional regression|Predict blood pressure, heart rate, and cholesterol levels for an individual.|

|   |   |
|---|---|
**Table 2.** Generative AI
|Model type|Example output|
|Text|Summarize an article.<br><br>  <br><br>Reply to customer reviews.<br><br>  <br><br>Translate documents from English to Mandarin.<br><br>  <br><br>Write product descriptions.<br><br>  <br><br>Analyze legal documents.|
|Image|Produce marketing images.<br><br>  <br><br>Apply visual effects to photos.<br><br>  <br><br>Generate product design variations.|
|Audio|Generate dialogue in a specific accent.<br><br>  <br><br>Generate a short musical composition in a specific genre, like jazz.|
|Video|Generate realistic-looking videos.<br><br>  <br><br>Analyze video footage and apply visual effects.|
|Multimodal|Produce multiple types of output, like a video with text captions.|

### Classification

A [**classification model**](https://developers.google.com/machine-learning/glossary#classification-model) predicts what category the input data belongs to, for example, whether an input should be classified as A, B, or C.

![A classification model is making predictions.](https://developers.google.com/static/machine-learning/problem-framing/images/classification-model.png)

**Figure 1.** A classification model making predictions.

Based on the model's prediction, your app might make a decision. For example, if the prediction is category A, then do X; if the prediction is category B, then do, Y; if the prediction is category C, then do Z. In some cases, the prediction _is_ the app's output.

![The product code uses model's output to make a decision.](https://developers.google.com/static/machine-learning/problem-framing/images/class-product-code.png)

**Figure 2.** A classification model's output being used in the product code to make a decision.

### Regression

A [**regression model**](https://developers.google.com/machine-learning/glossary#regression-model) predicts a numerical value.

![A regression model is making a prediction.](https://developers.google.com/static/machine-learning/problem-framing/images/regression-model.png)

**Figure 3.** A regression model making a numeric prediction.

Based on the model's prediction, your app might make a decision. For example, if the prediction falls within range A, do X; if the prediction falls within range B, do Y; if the prediction falls within range C, do Z. In some cases, the prediction _is_ the app's output.

![The product code uses the model's output to make a decision.](https://developers.google.com/static/machine-learning/problem-framing/images/regression-decision.png)

**Figure 4.** A regression model's output being used in the product code to make a decision.

Consider the following scenario:

You want to [cache](https://wikipedia.org/wiki/Cache_(computing)) videos based on their predicted popularity. In other words, if your model predicts that a video will be popular, you want to quickly serve it to users. To do so, you'll use the more effective and expensive cache. For other videos, you'll use a different cache. Your caching criteria is the following:

- If a video is predicted to get 50 or more views, you'll use the expensive cache.
- If a video is predicted to get between 30 and 50 views, you'll use the cheap cache.
- If the video is predicted to get less than 30 views, you won't cache the video.

You think a regression model is the right approach because you'll be predicting a numeric value—the number of views. However, when training the regression model, you realize that it produces the same [loss](https://developers.google.com/machine-learning/glossary#loss-function) for a prediction of 28 and 32 for videos that have 30 views. In other words, although your app will have very different behavior if the prediction is 28 versus 32, the model considers both predictions equally good.

![A model being trained and its loss evaluated.](https://developers.google.com/static/machine-learning/problem-framing/images/training.png)

**Figure 5.** Training a regression model.

Regression models are unaware of product-defined thresholds. Therefore, if your app's behavior changes significantly because of small differences in a regression model's predictions, you should consider implementing a classification model instead.

In this scenario, a classification model would produce the correct behavior because a classification model would produce a higher loss for a prediction of 28 than 32. In a sense, classification models produce thresholds by default.

This scenario highlights two important points:

- **Predict the decision**. When possible, predict the decision your app will take. In the video example, a classification model would predict the decision if the categories it classified videos into were "no cache," "cheap cache," and "expensive cache." Hiding your app's behavior from the model can cause your app to produce the wrong behavior.
    
- **Understand the problem's constraints**. If your app takes different actions based on different thresholds, determine if those thresholds are fixed or dynamic.
    
    - Dynamic thresholds: If thresholds are dynamic, use a regression model and set the thresholds limits in your app's code. This lets you easily update the thresholds while still having the model make reasonable predictions.
    - Fixed thresholds: If thresholds are fixed, use a classification model and label your datasets based on the threshold limits.
    
    In general, most cache provisioning is dynamic and the thresholds change over time. Therefore, because this is specifically a caching problem, a regression model is the best choice. However, for many problems, the thresholds will be fixed, making a classification model the best solution.
    

Let's take a look at another example. If you're building a weather app whose ideal outcome is to tell users how much it will rain in the next six hours, you could use a regression model that predicts the label `precipitation_amount.`

|**Ideal outcome**|**Ideal label**|
|---|---|
|Tell users how much it will rain in their area in the next six hours.|`precipitation_amount`|

In the weather app example, the label directly addresses the ideal outcome. However, in some cases, a one-to-one relationship isn't apparent between the ideal outcome and the label. For example, in the video app, the ideal outcome is to recommend useful videos. However, there's no label in the dataset called `useful_to_user.`

|**Ideal outcome**|**Ideal label**|
|---|---|
|_Recommend useful videos._|`?`|

Therefore, you must find a proxy label.


### Proxy labels

[**Proxy labels**](https://developers.google.com/machine-learning/glossary#proxy-labels) substitute for labels that aren't in the dataset. Proxy labels are necessary when you can't directly measure what you want to predict. In the video app, we can't directly measure whether or not a user will find a video useful. It would be great if the dataset had a `useful` feature, and users marked all the videos that they found useful, but because the dataset doesn't, we'll need a proxy label that substitutes for usefulness.

A proxy label for usefulness might be whether or not the user will share or like the video.

|**Ideal outcome**|**Proxy label**|
|---|---|
|_Recommend useful videos._|`shared OR liked`|

Be cautious with proxy labels because they don't directly measure what you want to predict. For example, the following table outlines issues with potential proxy labels for _Recommend useful videos_:

|**Proxy label**|**Issue**|
|---|---|
|Predict whether the user will click the "like" button.|Most users never click "like."|
|Predict whether a video will be popular.|Not personalized. Some users might not like popular videos.|
|Predict whether the user will share the video.|Some users don't share videos. Sometimes, people share videos because they **don't** like them.|
|Predict whether the user will click play.|Maximizes clickbait.|
|Predict how long they watch the video.|Favors long videos differentially over short videos.|
|Predict how many times the user will rewatch the video.|Favors "rewatchable" videos over video genres that aren't rewatchable.|

No proxy label can be a perfect substitute for your ideal outcome. All will have potential problems. Pick the one that has the least problems for your use case.




### Generation

In most cases, you won't train your own generative model because doing so requires massive amounts of training data and computational resources. Instead, you'll customize a pre-trained generative model. To get a generative model to produce your desired output, you might need to use one or more of the following techniques:

- [**Distillation**](https://developers.google.com/machine-learning/glossary#distillation). To create a smaller version of a larger model, you generate a synthetic labeled dataset from the larger model that you use to train the smaller model. Generative models are typically gigantic and consume substantial resources (like memory and electricity). Distillation allows the smaller, less resource-intensive model to approximate the performance of the larger model.
    
- [**Fine-tuning**](https://developers.google.com/machine-learning/glossary#fine-tuning) or [**parameter-efficient tuning**](https://developers.google.com/machine-learning/glossary#parameter-efficient-tuning). To improve the performance of a model on a specific task, you need to further train the model on a dataset that contains examples of the type of output you want to produce.
    
- [**Prompt engineering**](https://developers.google.com/machine-learning/glossary#prompt-engineering). To get the model to perform a specific task or produce output in a specific format, you tell the model the task you want it to do or explain how you want the output formatted. In other words, the prompt can include natural language instructions for how to perform the task or illustrative examples with the desired outputs.
    
    For example, if you want short summaries of articles, you might input the following:
    
    ```
    Produce 100-word summaries for each article.
    ```
    
    If you want the model to generate text for a specific reading level, you might input the following:
    
    ```
    All the output should be at a reading level for a 12-year-old.
    ```
    
    If you want the model to provide its output in a specific format, you might explain how the output should be formatted—for example, "format the results in a table"—or you could demonstrate the task by giving it examples. For instance, you could input the following:
    
    ```
    Translate words from English to Spanish.
    
    English: Car
    Spanish: Auto
    
    English: Airplane
    Spanish: Avión
    
    English: Home
    Spanish:______
    ```
    

Distillation and fine-tuning update the model's [**parameters**](https://developers.google.com/machine-learning/glossary#parameter). Prompt engineering doesn't update the model's parameters. Instead, prompt engineering helps the model learn how to produce a desired output from the context of the prompt.

In some cases, you'll also need a [**test dataset**](https://developers.google.com/machine-learning/glossary#test-set) to evaluate a generative model's output against known values, for example, checking that the model's summaries are similar to human-generated ones, or that humans rate the model's summaries as good.

Generative AI can also be used to implement a predictive ML solution, like classification or regression. For example, because of their deep knowledge of natural language, [**large language models (LLMs)**](https://developers.google.com/machine-learning/glossary#large-language-model) can frequently perform text classification tasks better than predictive ML trained for the specific task.


## Define the success metrics

Define the metrics you'll use to determine whether or not the ML implementation is successful. Success metrics define what you care about, like engagement or helping users take appropriate action, such as watching videos that they'll find useful. Success metrics differ from the model's evaluation metrics, like [**accuracy**](https://developers.google.com/machine-learning/glossary#accuracy), [**precision**](https://developers.google.com/machine-learning/glossary#precision), [**recall**](https://developers.google.com/machine-learning/glossary#recall), or [**AUC**](https://developers.google.com/machine-learning/glossary#auc-area-under-the-roc-curve).

For example, the weather app's success and failure metrics might be defined as the following:

|**Success**|Users open the "Will it rain?" feature 50 percent more often than they did before.|
|---|---|
|**Failure**|Users open the "Will it rain?" feature no more often than before.|

The video app metrics might be defined as the following:

|**Success**|Users spend on average 20 percent more time on the site.|
|---|---|
|**Failure**|Users spend on average no more time on site than before.|

We recommend defining ambitious success metrics. High ambitions can cause gaps between success and failure though. For example, users spending on average 10 percent more time on the site than before is neither success nor failure. The undefined gap is not what's important.






What's important is your model's capacity to move closer—or exceed—the definition of success. For instance, when analyzing the model's performance, consider the following question: Would improving the model get you closer to your defined success criteria? For example, a model might have great evaluation metrics, but not move you closer to your success criteria, indicating that even with a perfect model, you wouldn't meet the success criteria you defined. On the other hand, a model might have poor evaluation metrics, but get you closer to your success criteria, indicating that improving the model would get you closer to success.

The following are dimensions to consider when determining if the model is worth improving:

- **Not good enough, but continue**. The model shouldn't be used in a production environment, but over time it might be significantly improved.
    
- **Good enough, and continue**. The model could be used in a production environment, and it might be further improved.
    
- **Good enough, but can't be made better**. The model is in a production environment, but it is probably as good as it can be.
    
- **Not good enough, and never will be**. The model shouldn't be used in a production environment and no amount of training will probably get it there.
    

When deciding to improve the model, re-evaluate if the increase in resources, like engineering time and compute costs, justify the predicted improvement of the model.

After defining the success and failure metrics, you need to determine how often you'll measure them. For instance, you could measure your success metrics six days, six weeks, or six months after implementing the system.

When analyzing failure metrics, try to determine why the system failed. For example, the model might be predicting which videos users will click, but the model might start recommending clickbait titles that cause user engagement to drop off. In the weather app example, the model might accurately predict when it will rain but for too large of a geographic region.



# Implementing a model


When implementing a model, start simple. Most of the work in ML is on the data side, so getting a full pipeline running for a complex model is harder than iterating on the model itself. After setting up your data pipeline and implementing a simple model that uses a few features, you can iterate on creating a better model.

Simple models provide a good baseline, even if you don't end up launching them. In fact, using a simple model is probably better than you think. **Starting simple helps you determine whether or not a complex model is even justified.**



## Train your own model versus using an already trained model

Trained models exist for a variety of use cases and offer many advantages. However, trained models only really work when the label and features match your dataset exactly. For example, if a trained model uses 25 features and your dataset only includes 24 of them, the trained model will most likely make bad predictions.

Commonly, ML practitioners use matching subsections of inputs from a a trained model for fine-tuning or transfer learning. If a trained model doesn't exist for your particular use case, consider using subsections from a trained model when training your own.


## Monitoring

During problem framing, consider the monitoring and alerting infrastructure your ML solution needs.

### Model deployment

In some cases, a newly trained model might be worse than the model currently in production. If it is, you'll want to prevent it from being released into production and get an alert that your automated deployment has failed.

### Training-serving skew

If any of the incoming features used for inference have values that fall outside the distribution range of the data used in training, you'll want to be alerted because it's likely the model will make poor predictions. For example, if your model was trained to predict temperatures for equatorial cities at sea level, then your serving system should alert you of incoming data with latitudes and longitudes, and/or altitudes outside the range the model was trained on. Conversely, the serving system should alert you if the model is making predictions that are outside the distribution range that was seen during training.

### Inference server

If you're providing inferences through an RPC system, you'll want to monitor the RPC server itself and get an alert if it stops providing inferences.


# Summary

 bookmark_border

Framing a problem in terms of ML is a two-step process:

1. Verify that ML is a good approach by doing the following:
    
    - Understand the problem.
    - Identify a clear use case.
    - Understand the data.
2. Frame the problem in ML terms by doing the following:
    
    - Define the ideal outcome and the model's goal.
    - Identify the model's output.
    - Define success metrics.

These steps can save time and resources by setting clear goals and providing a shared framework for working with other ML practitioners.





