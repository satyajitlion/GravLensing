tag: #MockLens #MockLensGeneration #TrainingData
### Today's Focus

Create a GitHub repository for the research. Convert the mock lens generation code from Jupyter notebook format to an executable Python script for more efficient large-scale processing. Use obsidian to document the research through research logs. 

### What I was able to accomplish

I created a GitHub repository containing the research from May and uploaded all my prior work to GitHub. I then worked toward creating the python files named ```constants.py``` and ```generateMockLens.py```. The former contained the constants from the original jupyter notebook file while the latter contained the function used to generate the mock lenses given the constants, which I copy pasted from the original notebook into the python scripts individually. Running the code, I then checked to see if the script ran fine and if there were any errors in mock lens generation.  

### Results

Given the time it took to generate the mock lenses, I reduced the number of mock lenses first ```num_mock = 10```, which took approximately 5.7766 seconds to run. I then increased the number to ```num_mock = 30``` which took approximately $177.29$ seconds to run the code. Given this rate, for $10^5$ mock lenses, this would approximately take 6-7 days to generate all of the mock lenses. 


### Problems

### Questions

### Next Steps