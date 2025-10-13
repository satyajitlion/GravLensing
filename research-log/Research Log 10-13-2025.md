### Today's Focus

Understanding and documenting the critical concepts of model evaluation in machine learning, specifically the creation and purpose of validation sets using the Keras API, and the important distinction between validation and test datasets.
***
### What I was able to accomplish

- Researched and synthesized the methodology for building a validation dataset in Keras via two methods: explicit `validation_data` and implicit `validation_split`.
    
- Documented a crucial implementation detail: `validation_split` takes the last portion of the data by default, emphasizing the necessity of shuffling (`shuffle=True`) to prevent bias.
    
- Established a key performance metric: validation accuracy should be within ~1-2% of training accuracy to indicate a well-fitted model.
    
- Clearly defined the conceptual and practical difference between a **validation set** (used during training for evaluation and tuning) and a **test set** (used only after training is complete for final evaluation on unseen data).
***
### Results

I now have a clear, practical guide in my notes for implementing model validation in my future Keras models. The most significant result was the conceptual clarification: the validation set is a "simulated test" during training, while the true test set is completely unseen data. This directly informs my research plan: the SIS+shear dataset will be split for training/validation, and the final model will be evaluated on the completely separate SIE dataset as the true test set.
***
### Challenges & Pause Points

- I had to pause to fully grasp why `validation_split` taking the last segment of data was problematic. I worked through the example array `[1, 2, 3, ..., 10]` to visualize the bias that would be introduced if the data was ordered.
    
- The distinction between "validation" and "test" was initially subtle. I had to consult multiple sources (video, documentation) to solidify my understanding that validation is for _model development_ and test is for _model assessment_.
***
### Questions & Ideas

- For my specific research data (SIS+shear), is it better to use `validation_split` or manually create a `validation_data` set? Would manual creation give me more control, for example, to ensure the validation set is representative of the entire data distribution?

	- When I implement this, I should write a function to plot training accuracy/loss and validation accuracy/loss on the same graph over each epoch. This will make it visually immediate to spot overfitting/underfitting.
	
- From what I have learned, I know that validation accuracy should be close to training accuracy. What if my validation accuracy is consistently _higher_ than my training accuracy as was seen in the video? Is this indicative of something? Might need to learn more about this. 
***
### Next Steps



##### Tags: #Keras #DeepLearning #DataPreprocessing #Foundations #NeuralNetworks #SequentialModel #TrainingData #TestData #Python 




