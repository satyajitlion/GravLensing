### Today's Focus

Addressing severe overfitting in the sequential neural network for gravitational lensing parameter estimation and implementing data normalization to handle a variable numbers of images per lens system. The goal was to improve cross-model generalization from SIS+shear training data to SIE and SIE+shear test datasets.
***
### What I was able to accomplish

- Diagnosed and fixed array dimension mismatches in the data extraction pipeline by implementing consistent reshaping of time delay and potential arrays across systems with 1-4 images.
    
- Enhanced the sequential network architecture with regularization: L2 weight decay (`0.001`), Dropout layers (`0.3 rate`), and Batch Normalization after each dense layer.
    
- Implemented training controls including Early Stopping (`patience=15`) and `ReduceLROnPlateau` scheduling to prevent overfitting and stabilize convergence.
    
- Attempted to optimize the data normalization script by switching from inefficient list operations to numpy concatenation. However, this seemed to have little effect on the time it took to complete the operation.
    
- Created a scaler saving system that ensures consistent normalization across training and test datasets for fair evaluation.
***
### Results

- Training now automatically stops at optimal epochs (around 30 instead of running to 100), preventing overfitting while maintaining learning efficiency.
    
- Learning curves show parallel, smooth convergence between training and validation loss, indicating the network is learning generalizable patterns rather than memorizing.
    
- Reduced the train-test performance gap from 30-50x to 6-7x: SIE test MAE improved to 0.1612 and SIE+shear test MAE to 0.1767, compared to training MAE of ~0.01. This is very impactful as the datasets are completely different to begin with. However, I used the same scalers to normalize training and test datasets which could have impacted the result and for this, I'll need to read more about it to make sure it doesn't add any additional bias to the network.
    
- The sequential network now demonstrates meaningful cross-model generalization (SIS+shear $\rightarrow$ SIE and SIS+shear $\rightarrow$ SIE+shear) which looks promising.
    
- Data preprocessing correctly handles the natural variation in lens systems (1-4 images) without dimension errors or performance penalties.
***
### Challenges & Pause Points

- Initial normalization implementation caused severe performance degradation due to Python list operations (still not yet resolved).
    
- Array shape inconsistencies between systems with different image counts required systematic handling using `np.atleast_1d()` and `.reshape(-1, 1)` for consistent 2D arrays.
    
- Import conflicts emerged when modularizing the normalization code, requiring careful debugging of function names and file structure. Therefore, the entire day was primarily spent debugging and there are still potential "hidden" errors within the code that I need to check very carefully for.
    
- The remaining 6-7x performance gap between training and cross-model testing suggests inherent differences in SIS+shear vs SIE or SIS+shear vs SIE+shear physical parameter spaces. 
    
- Is the current sequential architecture with improved regularization is sufficient or if more fundamental changes (different network types such as the GCNN) are needed for further improvements.
---
### Questions & Ideas

- Given the improved but still significant cross-model gap, would switching to a different architecture (CNNs, GCNNs, or transformers) provide substantial benefits over the current regularized sequential network?
    
- Could embedding physical constraints directly into the loss function (enforcing lens equation relationships) further improve generalization without architectural changes? I know the answer to this but still thought it was good to think about (the answer, I presume, to be "no" as this would introduce bias into the network which isn't what we want).
    
- How effective would mixed training data (SIS+shear + SIE + SIE+shear) be compared to the current SIS+shear only training approach? Would this generalize better? 
    
- Is the remaining performance gap acceptable for practical applications, or should we prioritize architectural experimentation over further tuning of the current sequential model?
---
### Next Steps

1. Keep debugging the code.

2. Continue learning more about GCNNs and consider if switching to a GCNN would be beneficial. Maybe there might be some value in directly comparing and contrasting?

3. Test the current sequential network on real gravitational lens system. Do the same for the future GCNN network. Which one is better?
    
4. Ask about mixing the different datasets and if that would yield better generalizability. If this were to be done, what would be the test data set?
        
5. If the sequential network proves to be insufficient, switch to different network like the GCNN.
##### Tags: #Sequential_Network #Overfitting_Fix #Regularization #Data_Normalization #Cross_Model_Generalization #Lensing_Physics #Neural_Networks



