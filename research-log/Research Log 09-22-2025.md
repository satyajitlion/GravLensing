### Today's Focus

Create a GitHub repository for the research. Convert the mock lens generation code from Jupyter notebook format to an executable Python script for more efficient large-scale processing. Use obsidian to document the research through research logs. 

***
### What I was able to accomplish

I created a GitHub repository containing the research from May and uploaded all my prior work to GitHub. I then worked toward creating the python files named ```constants.py``` and ```generateMockLens.py```. The former contained the constants from the original Jupyter notebook file while the latter contained the function used to generate the mock lenses given the constants, which I copy pasted from the original notebook into the python scripts individually. Running the code, I then checked to see if the script ran fine and if there were any errors in mock lens generation. I then tested for the amount of time it took to run the script and tried to streamline the code to minimize the time it took to run. Following this, I worked towards testing the generated mock lenses to see if there were any errors within the graphical representation of the lenses. This is where I had to increase the number of mock lenses from 10 to 30 to run my ```Analysis.ipynb``` notebook without errors. I did so because previously, I had picked Mock Lens number 26 from the synthetic dataset to graph the gravitational lens in order to spot any errors. Following this, I skimmed through the internet to find a way to connect obsidian to GitHub so that I could document my research progress through research logs, which I was able to do. I then worked towards setting up obsidian so I can stream line research-log generation. I additionally also created a ```README.md``` file to provide some context for the research and information. 

***
### Results

Given the time it took to generate the mock lenses, I reduced the number of mock lenses first ```num_mock``` = $10$, which took approximately 5.7766 seconds to run. I then increased the number to ```num_mock``` = $30$ which took approximately $177.29$ seconds to run the code. Given this rate, for $10^5$ mock lenses, this would approximately take 6-7 days to generate all of the mock lenses. 

***
### Problems

The required number of Mock lenses needed to train a network hovers around ```num_mock```$= 10^5$. Given the rate at which my computer generates the lenses, the total time to generate the lenses would then hover around 6-7 days of constant computation. This is troublesome as my cpu power is quite limited and cannot handle that kind of computation 

***
### Questions & Ideas

Amarel, the Rutgers super computer, might simplify the above issue and make the computation task simpler/easier. 

***
### Next Steps

Study how Amarel works and go back to notes from Computational Astro to learn how to login to Amarel and how to provide it with executable scripts. There might be a queue time which might extend the time it takes to execute the file. Learn about parallel processing to split the script into several pieces that can simultaneously run to conserve some time (useful for when ```num_mock``` $\geq 10^6$ or $10^7$). Additionally, consider how to describe the system using a "metric" such that when the system undergoes rotation, the key characteristics of the system would be the same. How would a network look like when trained on such data? Might need to research more on Tensors and how they work computationally. 

##### Tags: #MockLens #MockLensGeneration #TrainingData