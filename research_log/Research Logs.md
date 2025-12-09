***
## Research Log 09-22-2025
### Today's Focus

Create a GitHub repository for the research. Convert the mock lens generation code from Jupyter notebook format to an executable Python script for more efficient large-scale processing. Use obsidian to document the research through research logs. 
### What I was able to accomplish

I created a GitHub repository containing the research from May and uploaded all my prior work to GitHub. I then worked toward creating the python files named ```constants.py``` and ```generateMockLens.py```. The former contained the constants from the original Jupyter notebook file while the latter contained the function used to generate the mock lenses given the constants, which I copy pasted from the original notebook into the python scripts individually. Running the code, I then checked to see if the script ran fine and if there were any errors in mock lens generation. I then tested for the amount of time it took to run the script and tried to streamline the code to minimize the time it took to run. Following this, I worked towards testing the generated mock lenses to see if there were any errors within the graphical representation of the lenses. This is where I had to increase the number of mock lenses from 10 to 30 to run my ```Analysis.ipynb``` notebook without errors. I did so because previously, I had picked Mock Lens number 26 from the synthetic dataset to graph the gravitational lens in order to spot any errors. Following this, I skimmed through the internet to find a way to connect obsidian to GitHub so that I could document my research progress through research logs, which I was able to do. I then worked towards setting up obsidian so I can stream line research-log generation. I additionally also created a ```README.md``` file to provide some context for the research and information. 
### Results

Given the time it took to generate the mock lenses, I reduced the number of mock lenses first ```num_mock``` = $10$, which took approximately 5.7766 seconds to run. I then increased the number to ```num_mock``` = $30$ which took approximately $177.29$ seconds to run the code. Given this rate, for $10^5$ mock lenses, this would approximately take 6-7 days to generate all of the mock lenses. 
### Challenges & Pause Points

The required number of Mock lenses needed to train a network hovers around ```num_mock```$= 10^5$. Given the rate at which my computer generates the lenses, the total time to generate the lenses would then hover around 6-7 days of constant computation. This is troublesome as my cpu power is quite limited and cannot handle that kind of computation 
### Questions & Ideas

Amarel, the Rutgers super computer, might simplify the above issue and make the computation task simpler/easier. 
### Next Steps

Study how Amarel works and go back to notes from Computational Astro to learn how to login to Amarel and how to provide it with executable scripts. There might be a queue time which might extend the time it takes to execute the file. Learn about parallel processing to split the script into several pieces that can simultaneously run to conserve some time (useful for when ```num_mock``` $\geq 10^6$ or $10^7$). Additionally, consider how to describe the system using a "metric" such that when the system undergoes rotation, the key characteristics of the system would be the same. How would a network look like when trained on such data? Might need to research more on Tensors and how they work computationally. 

***
## Research Log 09-23-2025

### Today's Focus

Establish the foundational setup required to use the Amarel cluster. The primary objectives were to understand the cluster's operation, install necessary software (PuTTy, WinSCP, Emacs), and establish a connection method. The ultimate goal was to log in, create a project directory for the gravitational lensing research, and transfer the initial set of files.
### What I was able to accomplish

- **Knowledge Acquisition:** Spent significant time (5-6 hours) reviewing Computational Astrophysics course materials and OARC documentation to compile essential notes on SLURM workload manager commands and High-Performance Computing (HPC) principles specific to Amarel.    
- **Software Setup:** Successfully downloaded and installed the core software toolkit for cluster interaction:
    - **Emacs:** For efficient text editing directly on the cluster.
    - **WinSCP:** For secure graphical file transfer between my local machine and Amarel.
    - **PuTTy:** As a backup SSH client for terminal access.
- **Research & Evaluation:** Researched alternative tools like MobaXterm and WSL to ensure a robust setup, deciding on the aforementioned software stack for initial attempts.

This day was dedicated to preparation, building the knowledge base and local tooling necessary for effective interaction with the HPC environment.
### Results

The day's work resulted in a prepared local machine with the correct software and a refreshed understanding of HPC concepts and SLURM commands. However, the critical step of establishing a connection to Amarel was blocked by a network access issue.
### Challenges & Pause Points

The primary obstacle is **network connectivity**. To access Amarel from off-campus, a VPN connection to the Rutgers network is required. The recommended software, Cisco AnyConnect, cannot be installed due to a permission error. This prevents the use of both WinSCP and PuTTy, halting progress on file transfer and cluster login.
### Questions & Ideas

- **VPN Access:** Is the Cisco AnyConnect software freely available through a different Rutgers portal that I may have missed? Are there any alternative, university-approved VPN solutions for student researchers?
- **Connection Necessity:** Is PuTTy strictly necessary, or can a modern terminal like Git Bash handle the SSH connection directly, potentially simplifying the setup?
- **Access Paths:** Given the VPN issue, should the immediate plan shift to gaining physical access to a campus location with a direct Rutgers network connection, such as Room 330 in the Serin Physics Building?
### Next Steps

1. **Resolve Connectivity:** The top priority is to solve the VPN issue. I will search for alternative installation paths for Cisco AnyConnect or contact Rutgers OIT support to resolve the permission error.    
2. **Seek Physical Access:** If the VPN cannot be configured immediately, I will confirm my access privileges to Room 330 or the undergraduate lounge to work directly from a campus machine on the Rutgers network.
3. **Confirm Amarel Access:** Verify that my Amarel account is active and in good standing. If not, I will contact Dr. Keeton for sponsorship.
4. **Execute File Transfer Plan:** Once connected, the immediate technical steps are to create the `GravLensing` subdirectory in my Amarel home directory, transfer the Python scripts via WinSCP, and begin composing the SLURM job script.

***
## Research Log 09-24-2025
### Today's Focus

Dedicate time to understanding the theoretical foundations of neural networks and understand how Keras works. The goal was to understand the mathematics behind NNs by deriving the core equations for a standard, hand-coded Artificial Neural Network (ANN), focusing on forward propagation, loss functions, and the principles of backpropagation.
### What I was able to accomplish

I made significant progress in building a fundamental understanding of neural network mechanics. The primary output was the creation of detailed notes in the document [Neural Networks and Keras Notes](https://github.com/satyajitlion/GravLensing/blob/c45bbfe521683355478edba71deb8bf29333cdb6/Notes/Neural%20Networks%20and%20Keras%20Notes.md) My work involved:

- **Mathematical Derivation:** I systematically worked through the mathematical transformations that occur in an ANN, starting from the input layer through to the hidden layers. This involved deriving how input data is transformed via weight matrices and bias vectors to produce an output.
- **Core Concepts Documented:** I documented key concepts including the role of loss functions for both individual samples and batches, the critical importance of the learning rate ($\alpha$) as a hyperparameter, and the challenges of model fitting (underfitting, overfitting, and the ideal best-fit).    
- **Practical Implementation:** To solidify my understanding, I followed a tutorial to implement a basic neural network using only NumPy, without the abstraction of a high-level framework like Keras. This hands-on exercise highlighted the complexity involved in manual implementation and underscored the utility of libraries like Keras for streamlining the process.

This deep dive provided a strong conceptual foundation that will be crucial for effectively designing and troubleshooting neural networks later in the project.
### Questions & Ideas

- **Architectural Difference:** How do Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) fundamentally differ from the standard ANN I studied today? Which architecture is most suitable for analyzing 2D gravitational lensing images?
- **Invariance in the Model:** As suggested by Dr. Keeton, a major consideration is how to design the network's input or structure to be invariant to rotations and translations. The model should recognize that a rotated lens is the same underlying system. What is the best way to engineer this? Should the input data be pre-processed into a rotation-invariant metric, or should the network architecture itself (perhaps using a CNN) inherently handle these symmetries?
### Next Steps

1. **Continue Keras/TensorFlow Learning:** If physical access to campus workstations is not possible, the next step is to watch the tutorial video [Keras With TensorFlow](https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=423s) to transition from theoretical math to practical implementation.
2. **Prioritize Mock Lens Generation:** If I can access Room 330 or the undergraduate lounge, the immediate priority is to resolve the Python dependency issues on Amarel and begin the large-scale generation of mock lenses, as this data is a prerequisite for training any neural network.

***
## Research Log 09-25-2025
### Today's Focus

Gain physical access to the research workstation (Room 330, Serin Physics Building) and establish a connection to the Amarel cluster. The primary objectives were to transfer project files to the cluster, create a dedicated project directory, and successfully submit a batch job via the SLURM workload manager to begin generating mock gravitational lenses.
### What I was able to accomplish

- **Access & Setup:** Successfully accessed the Amarel cluster via SSH directly from Git Bash, eliminating the need for PuTTy. Created a well-organized project structure by making a `GravLensing` subdirectory within my home directory.
- **File Management:** Used WinSCP to securely transfer all necessary Python scripts (`pygravlens.py`, `constants.py`, `generateMockLens.py`) from my local machine to the `GravLensing` directory on Amarel. Updated the `num_mock` parameter to $10^5$ in preparation for a large-scale run.
- **SLURM Job Submission:** After extensive consultation of OARC documentation and external resources to overcome a lack of course materials, I composed a functional SLURM script (`run_lens_job.slurm`). The script was configured with appropriate resource requests (24-hour walltime, 8 GB RAM) for the Amarel cluster. I submitted the job successfully using the `sbatch` command.

This process involved significant troubleshooting and independent learning to understand SLURM script syntax and Amarel's specific configuration, representing a major step forward in using HPC resources.
### Results

- Established a reliable workflow for accessing Amarel and managing project files.
- Created and submitted a SLURM batch job intended to generate 100,000 mock lenses.
- Identified a critical roadblock: the job failed immediately due to missing Python dependencies, halting progress on the primary objective.
### Challenges & Pause Points

The main obstacle encountered was a **dependency management issue**. The batch job failed because the required Python libraries (NumPy, Astropy, SciPy, Shapely, Matplotlib) were not installed in my Amarel environment.

A secondary issue was **Python version configuration**. The system default was Python 2.7.5, while my code requires Python 3.8.2. I had to manually locate and load the correct version using the `module` system, indicating my environment needs proper configuration for consistency.
### Questions & Ideas

- **Question:** What is the best practice for installing Python packages on Amarel? Should they be installed in my home directory, or is there a project-specific or shared location?
- **Idea:** While the lens generation job runs (which will take time), I can pivot to the next phase of the project: researching and prototyping the neural network component using Keras/TensorFlow.
- **Idea:** I need to conduct a comparative analysis of ANN, RNN, and CNN architectures to determine the most suitable model for analyzing the generated lensing images.
### Next Steps

1. **Resolve Dependencies:** Priority #1 is to correctly install the missing Python packages. I will consult OARC documentation on Python package management and ensure installations are done on a compute node, not the login node.
2. **Seek Guidance:** If I cannot resolve the installation issue quickly, I will consult with Dr. Burkhart or submit a ticket to the OARC help desk.
3. **Parallel Work:** Once the batch job is queued, I will begin studying neural network implementations with Keras to maintain project momentum.
4. **Architecture Research:** Initiate a literature review to decide on the optimal neural network architecture (ANN, RNN, CNN) for gravitational lens analysis.

***
## Research Log 09-26-2025
### Today's Focus

Read up more from the OARC's cluster resources and work towards learning how to fix yesterday's errors involving installation, the SLURM script, and incorrect usage of Amarel directory.
### What I was able to accomplish

Took notes on several key operational aspects of the Amarel cluster:

- **Software Access:** Learned the correct commands (`module use`, `module load`, `module spider`) to find and load community-contributed software, which was a source of yesterday's installation errors.
- **Proper Job Execution:** Understood the critical practice of using `srun` for interactive work or a SLURM script for batch jobs to run on compute nodes, not login nodes. This directly addresses the root cause of yesterday's errors.
- **Job Management:** Gathered essential commands for monitoring (`squeue`), canceling (`scancel`), and analyzing job efficiency (`sacct`) after a job completes.
- **Partition Types:** Clarified the different SLURM partitions (main, gpu, mem, nonpre, etc.), which is crucial for writing an effective SLURM script that requests the right resources.
- **Installation Best Practices:** Learned the correct method to install Python packages locally using `python -m pip install [package] --user` after loading the desired `python` module, which should resolve the installation issues.
- **File Management:** Picked up a useful command (`rm -f *.out *.err *.pyc`) for cleaning up output files from failed jobs.
### Challenges & Pause Points

- Yesterday's problems are now clearly identified as a combination of:
	1. **Incorrect Node Usage:** Running installation/compilation directly on the login node.
	2. **Improper Installation Method:** Not using the correct `pip` command with the `--user` flag after loading a module.
	3. **Potential SLURM Script Issues:** The script may have been submitted to the wrong partition or did not request appropriate resources.
### Questions & Ideas

- **Question:** For my specific software (e.g., a Python machine learning library with GPU support), which `python` module version and which `gpu` partition should I use? Should I use `nonpre` to avoid preemption for long jobs?
- **Idea:** Create a template SLURM script that includes headers for the `gpu` partition, requests a specific GPU type, loads necessary modules, and uses the correct `pip` installation method within the script if needed.
- **Idea:** Test the installation process for my required packages by first getting an interactive session on a compute node with `srun --pty -p gpu -t 00:30:00 --gres=gpu:1 /bin/bash` and then running the installation commands there. This isolates any potential issues from the login node.
### Next Steps

1. **Test Installation Interactively:** Use `srun` to get an interactive session on a GPU compute node.
2. **Install Software Correctly:** In the interactive session, load the appropriate `python` module and install the necessary packages using `python -m pip install [package] --user`.
3. **Revise SLURM Script:** Update my SLURM batch script based on the partition information. Key changes will include:
	- Specifying the correct partition (e.g., `#SBATCH -p gpu`).
	- Requesting GPU resources (e.g., `#SBATCH --gres=gpu:1`).
	- Adding commands to load the required modules within the script.
	- Ensuring the job executes in the correct directory (e.g., `/scratch/ssg131/`).
4. **Submit a Test Job:** Run a short, small-scale test job with the revised script to verify everything works.
5. **Monitor and Analyze:** Use `squeue` to monitor the job and `sacct` after completion to check resource usage (like MaxRSS) for efficiency.

***
## Research Log 09-29-2025
### Today's Focus

Revisit the foundational mathematical groundwork for understanding neural networks by reviewing core linear algebra and calculus concepts. Focus on matrix operations, gradient theory, and linear regression as the building blocks for more complex machine learning models. Work towards understanding the need for cost functions and why they are necessarily in a NN (answering questions from 09/24/2025). Begin the review of Logistic Regression.
### What I was able to accomplish

- Added clear notes on matrix shapes in NumPy, emphasizing the (rows, columns) convention and its critical importance in network layer calculations.
- Reviewed gradients, connecting the simple derivative $dy/dx$ to the multi-variable gradient $\nabla f$, highlighting its role in pointing in the direction of greatest increase.
- Thoroughly reviewed Linear Regression, breaking it down from simple to multivariable forms.
- Defined the Mean Squared Error (MSE) cost function and explained the intuition behind Gradient Descent for minimizing it, complete with a code snippet for calculating the gradient.
- Initiated the section on Logistic Regression (still need to finish the section).
### Results

- Successfully created a concise and clear reference for matrix operations and gradient theory.
- The Linear Regression review is well-documented with both mathematical formulas and practical Python code, reinforcing the connection between theory and implementation.
- The notes are now better structured for quick recall, which will be essential when implementing backpropagation in Neural Networks.
### Challenges & Pause Points

- The derivation of the gradient for the MSE cost function, while understood conceptually, can still be a bit tricky to manually execute without error. Need to practice this more.
- The Logistic Regression section is incomplete. The link between its cost function (log loss) and the Linear Regression cost function (MSE) needs to be clarified.
### Questions & Ideas

- For Logistic Regression, why can't we use MSE as the cost function? What property of the sigmoid activation function makes log loss more suitable?
- In my NumPy matrix examples, I see that arr1 with shape (1, 3) and arr2 with shape (3, 1) represent different mathematical objects. When we get to the forward propagation equation $z^{[1]} = W^{[1]}X + b^{[1]}$, how do I determine the correct dimensions for $W^{[1]}$ to ensure the matrix multiplication works and produces the output shape needed for the next layer? How would one ensure that the shapes are correctly matched such that matrix/vector multiplication is able to take place?
### Next Steps

1. If rooms are accessible, Generate Mock Lenses (priority above all else) and fix errors from Friday. 
2. Finish the notes on Logistic Regression, including its hypothesis (sigmoid), cost function (log loss/BCE), and why it's used for classification.
3. Explicitly link the completed Logistic Regression review to the Neural Network section, showing how a single neuron with a sigmoid activation is essentially a Logistic Regression unit.
4. Use the foundational knowledge from today to re-derive the backpropagation equations in the NN section step-by-step to ensure full understanding.

***
## Research Log 09-30-2025
### Today's Focus

Resolve the critical Python dependency issues on the Amarel cluster, specifically for the installation of the NumPy, SciPy, Astropy, Matplotlib, and shapely packages, to enable the successful execution of the mock lens generation SLURM batch job.
### What I was able to accomplish
- Discovered and loaded the `py-data-science-stack` module, which provided a pre-configured environment with Python and most required packages (NumPy, SciPy, Astropy, Matplotlib).
- Found the root cause: the system's Python 3.8.2 installation is fundamentally broken (missing `_ctypes` module), making scientific computing impossible without using specialized modules.
- Successfully resolved library conflicts by cleaning up mixed installations and ensuring all packages work within the `py-data-science-stack` environment.
- Finally achieved a stable, working Python environment where all dependencies (including the problematic `shapely` package) can be properly imported.
### Results

- The core issue was a broken system Python installation, not just missing packages. This explains why initial installation attempts failed.
- The `py-data-science-stack` module provides a properly configured environment for scientific computing on Amarel (with everything besides shapely).
- After cleaning up installation conflicts, most required packages were able to be imported (besides SciPy for Python 3.8.2 and Shapely for the built in python version given by the `py-data-science-stack` module).
### Challenges & Pause Points

- The system's Python 3.8.2 installation is missing critical components (`_ctypes` module), making it impossible to use scientific packages like SciPy. This explains why initial dependency installations failed. Additionally, my directory was using python version 2.7.5 or defaulting to it whenever I used to the `python [file_name]` command. 
- When switching to the working `py-data-science-stack` module, there were conflicts with previously installed packages that had broken library links.
- Required significant troubleshooting to clean up mixed installations and ensure consistency between interactive sessions and batch job environments.
- Shapely is still failing for the `py-data-science-stack` module which is why I am still unable to run my batch job without errors.
### Next Steps

- Reach out the Amarel support and explain the persisting error or leave this on pause for a bit?
- Immediately resubmit the mock lens generation batch job to begin producing the 100,000 lenses after installation related errors get fixed.
- Use `squeue` and `sacct` to track the job's status and resource usage.
- Start investigating ANN, RNN, and CNN architectures for the next phase while the batch job runs.
- Record the successful environment setup process for future reference and reproducibility.

***
## Research Log 10-01-2025 to 10-02-2025
### The Focus for these Two Days

To deepen my understanding of classification algorithms by reviewing Logistic Regression fundamentals and exploring different activation functions used in neural networks. The goal was to connect the concepts from traditional logistic regression to their applications in modern neural networks. 
### What I was able to accomplish

<u>tl;dr</u>: Edited [Neural Networks and Keras Notes](https://github.com/satyajitlion/GravLensing/blob/c45bbfe521683355478edba71deb8bf29333cdb6/Notes/Neural%20Networks%20and%20Keras%20Notes.md) and Researched the intricacies behind logistic regression and activation functions, their importance, and how it relates to a neural network that is attempting to learn non-linear patterns between it's inputs and outputs.

- **Completed Logistic Regression Review:** Thoroughly documented the theory behind logistic regression, including its use cases compared to linear regression.
- **Explained the Sigmoid Function:** Detailed how the sigmoid function transforms linear outputs into probabilities between 0 and 1.
- **Documented Cost Function:** Explained the cross-entropy/log loss function for logistic regression and how it handles binary classification.
- **Covered Probability Mapping:** Described how probabilities are converted to discrete classes using a threshold (typically 0.5).
- **Began Activation Function Exploration:** Started researching different types of activation functions, beginning with a deeper analysis of sigmoid's limitations and briefly introducing tanh and ReLU.
### Results

- Gained clarity on when to use logistic regression vs. linear regression through concrete examples (house prices vs. ticket classification).
- Understood the mathematical reasoning behind the cross-entropy cost function and how it elegantly handles both classes ($y=1$ and $y=0$) in one formula.
- Recognized that logistic regression can be seen as a single-layer neural network with a sigmoid activation function.
- Identified key limitations of the sigmoid function (vanishing gradients, slow convergence, non-zero-centered outputs) that motivate the use of other activation functions.
### Challenges & Pause Points

1. Understanding how logistic regression relates to neural networks required connecting the "sigmoid activation" concept across both topics.
2. The derivation and intuition behind the cross-entropy cost function is more complex than MSE for linear regression.
3. Recognizing why such a fundamental function has practical limitations in deep learning was initially counterintuitive.
### Questions & Ideas

1. If sigmoid has these limitations, why is it still commonly used in the output layer for binary classification?
2. How do tanh and ReLU specifically address the limitations of sigmoid? What are their own advantages and disadvantages?
### Next Steps

1. Learn more about the tanh and ReLU functions, including their mathematical properties, derivatives, and practical applications.
2. Explicitly map how logistic regression concepts translate to neural network layers and activation functions.
3. Maybe code a logistic regression model to apply these concepts hands-on.
4. Research how these binary classification concepts extend to multi-class problems using softmax activation.

***
## Research Log 10-03-2025
### Today's Focus

Understanding different types of activation functions in neural networks, their mathematical formulations, visual characteristics, advantages, and limitations - with particular attention to the vanishing gradient problem.
### What I was able to accomplish

- Studied and documented three main activation functions: Sigmoid, Tanh, and ReLU
- Compiled mathematical formulas for each activation function
- Understood the graphical representations and output ranges of each function
- Analyzed the specific problems associated with each activation function
- Gained a comprehensive understanding of the vanishing gradient problem through analogies
### Results

**Key Findings:**
- **Sigmoid**: Used for binary classification output layers, but suffers from vanishing gradients, slow convergence, and non-zero-centered output (range: 0-1)
- **Tanh**: Zero-centered improvement over sigmoid (range: -1 to 1), but still has vanishing gradient issues
- **ReLU**: Computationally efficient, avoids vanishing gradient problem, uses simple max(0,z) operation

**Note**: I discovered that as an activation function, ReLU is generally preferred over sigmoid and tanh due to its computational efficiency and ability to mitigate the vanishing gradient problem.
### Questions & Ideas

- Why exactly does ReLU avoid the vanishing gradient problem while sigmoid and tanh don't?
- Are there situations where sigmoid or tanh might still be preferable to ReLU?
- What about the "dying ReLU" problem I've heard mentioned?
- How do other activation functions (Leaky ReLU, ELU, Swish) compare to these three?
### Next Steps

1. Research and understand the "dying ReLU" problem and its solutions
2. Explore other modern activation functions beyond these three basic types
3. Get Started on Keras and learn how models are implemented through Keras
4. Fix Mock Lens Generation issues with Amarel (contact Amarel Support, touch up on this issue with Dr. Keeton)
5. Try to implement a small-scale example neural network model with Keras to test how model implementation works.

***
## Research Log 10-06-2025
### Today's Focus

Contact Amarel support at [help@oarc.rutgers.edu](https://mailto:help@oarc.rutgers.edu) to aid with installation help for external packages like shapely on the data science module. For the neural network aspect of the project, learn about the different regularization techniques that are used to avoid overfitting. Learn about the Keras implementation for the neural networks and the ways in which grad descent, nodes, activation functions, optimizers, and regularization is implemented.
### What I was able to accomplish

1. Composed and sent an email to OARC support ([help@oarc.rutgers.edu](https://mailto:help@oarc.rutgers.edu)) requesting assistance with installing the `shapely` package on the Amarel cluster.
2. Researched and documented the "Dying ReLU" problem, including its causes and solutions like Leaky ReLU.
3. Compiled a list of common activation functions (Binary Step, Parametric ReLU, ELU, Softmax, Swish, GELU, SELU) with their mathematical definitions and use-cases.
4. Began researching regularization techniques, completing notes on L1 and L2 regularization, including their cost functions and Keras implementations and started learning about the "Dropout" Regularization technique.
### Results

- **ReLU vs. Leaky ReLU:** Understood the critical failure mode of the standard ReLU function (Dying ReLU) and how Leaky ReLU provides a robust alternative by preventing zero gradients.
- **Activation Function Selection:** Gained a clearer understanding that activation function choice is problem-dependent, with ReLU being a common default for hidden layers and Softmax/Sigmoid being standard for classification output layers.
- **L1/L2 Regularization:** Confirmed that these techniques work by adding a penalty on the model's weights directly to the loss function, encouraging simpler models to reduce overfitting. L2 (weight decay) is more commonly used.
### Challenges & Pause Points

- The `shapely` package installation on Amarel is currently blocked, pending a response from OARC support.
- The research on regularization techniques is incomplete; notes on Dropout and Early Stopping need to be finished.
### Questions & Ideas

- What is the specific Keras syntax for implementing Dropout layers?
- In practice, is it better to use a single regularization technique or to combine them (e.g., L2 + Dropout)?
- When building the initial neural network model for this project, it might be good to start with ReLU activation in hidden layers (monitoring for signs of "dying" units) and use L2 regularization as a first step to combat overfitting. Also, I should use Softmax on the output layer for classification. I wonder if the Mock Lens data requires classification, however, as it's pretty clear that the data seems to be purely numerical. However, with regard to rotation of the gravitational lens system, I don't want a network that changes the output values simply as a result of rotation or translation of the system. It might be good to think about this for a while.
### Next Steps

1. Follow up on the OARC support ticket if no response is received within 1-2 business days.
2. Complete the research on regularization by finishing the notes on **Dropout** and **Early Stopping.**
3. Draft the initial architecture of the neural network using Keras, incorporating the insights on activation functions and L2 regularization.    
4. Implement and test the model on a small subset of data to verify the environment and basic functionality.

***
## Research Log 10-07-2025
### Today's Focus

Transition from theoretical neural network research back to the generation of Mock Gravitational Lenses using Amarel. Find the root cause of errors with Dr. Keeton and finish setting up the computational pipeline on Amarel cluster to generate mock lensing data for future neural network training, resolving environment configuration issues and establishing reliable batch processing workflows. Ensure that the local environment setup on Amarel is consistent in terms of the versions of it's installed modules and libraries. Finally, submit the batch job after fixing all the configuration related issues. Ensure that the batch job doesn't fail due to any other errors and double check the SLURM script to ensure that the batch job will run smoothly.
### What I was able to accomplish

- **Resolved critical NumPy compatibility issue** that was preventing pygravlens from running by creating a dedicated conda environment with compatible package versions (numpy=1.23.5, scipy=1.9.3)
- **Successfully configured the complete gravitational lensing simulation environment** with proper dependencies including shapely, astropy, matplotlib, scipy, and etc.
- **Implemented and tested the mock lens generation pipeline** with three mass model configurations in `generateMockLens.py`:
    - Shear-only models
    - Ellipticity-only models
    - Combined shear+ellipticity models
- **Successfully submitted the batch job** and am currently waiting for the batch job to finish over the next 24 hours.
### Results

- Created a stable, reproducible computational environment that resolves the dependency issues encountered over the course of these few weeks in Amarel (mainly issues with version control).
- Successfully submitted `mock_lens_100k` batch job generating three datasets of 100,000 gravitational lenses each
- Developed reliable process for job submission, monitoring, and management on HPC cluster
### Challenges & Pause Points

- **Initial Package Conflicts**: Spent significant time resolving NumPy/SciPy version incompatibilities that prevented pygravlens from importing
- **SLURM Learning Curve**: Required trial and error to properly configure batch scripts and understand job management commands
- **Editor Artifacts**: Encountered and resolved Emacs auto-save files (`#pygravlens.py#`) in working directory
- **Pending Completion**: The 100,000 lens simulation job is currently running - results and data quality verification are pending
### Questions & Ideas

- Should we implement data validation checks to ensure the generated mock lenses are physically realistic before using them for training?
- How do I ensure that for the Neural Network, translations and rotations of the system don't change the pattern that the network develops between the inputs and the outputs? 
	- <u>Idea</u>: Maybe use Tensors? Tensors have changing elements but a static metric that characterizes the system. But how do I do this? Will need to learn more here.
- <u>Idea</u>: Once data generation is verified, we could create a data loader that automatically feeds these .npy files into Keras for neural network training
### Next Steps

1. **Monitor Job Completion**: Track the `mock_lens_100k` job and verify successful generation of all three output files
2. **Data Validation**: Load and examine the generated datasets to check for physical consistency and data quality
3. **Statistical Analysis**: Analyze distributions of key observables (image positions, magnifications, time delays) across the three model types to ensure there were no further errors in Mock Lens Generation.
4. **Return to Neural Network Research**: Continue learning about regularization from yesterday and work towards developing a small scale Network model using Keras.
5. **Neural Network Preparation**: Begin designing the neural network architecture for this mock lens data. Think about how the network should be implemented, what type of network will be used for this data, etc. Also think about the Tensor issue and maintaining the characteristics of the mock gravitational lens system having undergone translation or rotation.    
6. **Pipeline Scaling**: Plan for larger simulation runs ($10^6$ lenses) once initial data quality is confirmed and also add the right hand side of the equation below to dictionary (the dictionary only contains the left hand side)

	```math 
	\left\{x_{i}, \Delta t_i\right\}, z_{l}, z_{s} \rightarrow 
	\left\{\phi_{i}, \alpha_{i}\right\}, D_t 
	```

***
## Research Log 10-08-2025
### Today's Focus

Continue neural network regularization research. Improve understanding of dropout and early stopping techniques, explore their practical implementations in Keras, and begin designing neural network architectures suitable for the mock lens data being generated. Bridge theoretical knowledge with practical application by considering how to handle geometric transformations in gravitational lens data.
### What I was able to accomplish

- Studyied dropout and early stopping techniques in depth to further deepen my understanding of regularization
- **Developed intuitive understanding of dropout** through real-world analogies and visual representations 
- **Documented practical code examples** for implementing both regularization techniques
- Learned about the **Early stopping implementation in Keras** with proper validation monitoring and patience parameters
### Challenges & Pause Points

- Need to figure out how I will apply these regularization techniques specifically to gravitational lens data once it is available.
- Still unclear how to ensure neural network predictions remain consistent under translations/rotations of lens systems (hypothetically, it might be sufficient to encode the lens equations as a tensor but I am unsure whether I can do this. Perhaps there is an easier way or a different way to ensure this?)
- Still unclear how I should determine optimal dropout rates and early stopping criteria for Grav Lens modeling if I proceed to use this regularization technique.
### Questions & Ideas

- How do I apply these regularization techniques specifically to gravitational lens data once available? Which regularization technique do I even use? Would using something like dropout regularization negatively affect the network as the size of the network is decreasing? If the size of the network is decreasing, bias will likely increase as a result. However, the size is only decreasing per layer and is still present for the output layer. 
- How to ensure neural network predictions are invariant to translations and rotations of the lens system?
	- _Idea_: Explore tensor-based approaches where the metric characteristics remain constant despite coordinate changes
	- _Idea_: Research geometric deep learning or equivariant neural networks for this specific challenge
- Once data generation is verified, create a data loader that automatically feeds .npy files into Keras
- Could dropout and early stopping work synergistically with other techniques like L2 regularization for our lens data?
### Next Steps

1. Track the `mock_lens_100k` job and verify successful generation of all three output files
2. Start working on building the neural network model for mock lens data, incorporating today's regularization learnings. Temporarily use the 30 mock lenses to create the model and ensure that the model can take in much larger datasets.
3. Investigate tensor methods and geometric deep learning for handling coordinate transformations
4. Develop checks to ensure physical consistency of generated lens data
5. Make sure to create some form of algorithm that is able to take in the dictionary I created and is able to unpack the information within said dictionary containing the mock lenses. Prepare for larger simulation runs ($10^5$ lenses) once initial data quality is confirmed. 
6. Add the right-hand side of the following equation to the data dictionary (maybe rerun the Amarel Batch job post edit to incorporate this change as well when generating the mock lenses?)

	```math
	\left\{x_{i}, \Delta t_i\right\}, z_{l}, z_{s} \rightarrow \left\{\phi_{i}, 
	\alpha_{i}\right\}, D_t
	```

***
## Research Log 10-09-2025 to 10-10-2025
### The Focus for these Two Days

Ensure the successful execution of the 100k-element mock lens generation batch job on the Amarel cluster while concurrently building foundational expertise in Keras to prepare for the machine learning phase of the project. This involves actively monitoring the job's status and resource consumption via SLURM commands and real-time log inspection, while systematically studying Keras from environment setup and core syntax to model architecture design principles and the end-to-end workflow for building and training neural networks. 
### What I was able to accomplish

- **Amarel Cluster Job:** Resubmitted the previous batch job which failed as a result of using more time than I had allotted. Began monitoring the 100k-element mock lens generation batch job. Researched how to make the batch job more efficient and am currently trying to figure out a way to implement parallel processing for the code in the case the next batch job fails.
- **Keras Foundation:** Established a local Keras/TensorFlow environment and progressed through the initial stages of the machine learning workflow:
    - Understood the core components of a `Sequential` model.
    - Researched and implemented data preparation techniques using `numpy` and `sklearn`, specifically focusing on converting data to arrays, shuffling, and normalization with `MinMaxScaler`.
    - Successfully coded the initial setup for a simple Sequential model, including imports for `Sequential`, `Dense`, `Activation`, and the `Adam` optimizer.
### Results

- The mock lens generation job is running on Amarel. Initial log inspection indicates it is proceeding without errors.
- Worked on developing a Keras model for sequential data and worked on data preparation & transformation in order to learn how the model matches the sizes of the datasets and uses it within its hidden layers. The data formatting is quite specific which is required for the `.fit()` function. Therefore, I worked on researching how I can actively used a 2D dataset (like mine) and flatten it such that it can serve as an input for the ML model without changing the key characteristic of the dataset. 
### Challenges & Pause Points

- The primary challenge encountered was a **conceptual gap in adapting the Keras Sequential model to my specific dataset structure.** My initial data preparation (creating `x` and `y` from dictionary keys/values) was correct for a simple numerical regression/classification problem. However, I paused when I realized my actual dataset for this research is more complex. The input `x` is not a single number but an array, and the output `y` is also an array likely containing text (e.g., multiple labels like redshift, image position, etc). The simple `Dense` layer example I started with assumes a flat input, not the multi-dimensional arrays I will be working with. Therefore, I am reconsidering using a sequential model and instead using a different model. I might need to ask Somayeh for help regarding this matter.
- The above "pause point" additionally has raised questions about the correct model architecture that I can use (e.g., using `Conv2D` layers for images, `Flatten` layers) and how to properly structure the input and output layers to handle this complexity.
### Questions & Ideas

- **Key Question:** How do I structure a Keras model when the input data is a multi-dimensional array (e.g., a 2D image of a lens) and the output is a set of multiple physical parameters?
- **Follow-up Questions:**
    - Does the `Sequential` API remain the best choice for this, or should I plan to use the Functional API for multi-input/multi-output models?
    - How do I specify the `input_shape` parameter for the first layer when my input is a 2D image?
    - For the final layer, how do I configure a `Dense` layer to predict, for example, three different parameters simultaneously? 
- The next phase of study should focus on Convolutional Neural Networks (CNNs) for image input and multi-output regression models. The Functional API might be necessary for the final model. Note that I should consult Somayeh for more help regarding this matter as she's more experienced with Machine Learning.
### Next Steps

1. **Deepen Keras Architecture Knowledge:** Shift focus from simple `Dense` networks to studying CNN architectures and the Functional API in Keras to handle image-like input and multi-parameter output.
2. **Design a Model Prototype:** Sketch out the architecture for a CNN that takes a 2D array (mock lens image) as input and outputs a vector of predicted parameters.
3. **Formalize Data Pipeline:** Plan how to load the 100k generated mock lenses from file into a data pipeline that can be fed into a Keras model.    
4. **Continue Cluster Monitoring:** Keep monitoring the Amarel job to completion (note that I allocated 48 hours to this batch job this time, therefore, I will have to check how the batch job is running the following Monday).

***
## Research Log 10-13-2025
### Today's Focus

Understanding and documenting the critical concepts of model evaluation in machine learning, specifically the creation and purpose of validation sets using the Keras API, and the important distinction between validation and test datasets.
### What I was able to accomplish

- Researched and synthesized the methodology for building a validation dataset in Keras via two methods: explicit `validation_data` and implicit `validation_split`.
- Documented a crucial implementation detail: `validation_split` takes the last portion of the data by default, emphasizing the necessity of shuffling (`shuffle=True`) to prevent bias.
- Established a key performance metric: validation accuracy should be within ~1-2% of training accuracy to indicate a well-fitted model.
- Clearly defined the conceptual and practical difference between a **validation set** (used during training for evaluation and tuning) and a **test set** (used only after training is complete for final evaluation on unseen data).
### Results

I now have a clear, practical guide in my notes for implementing model validation in my future Keras models. The most significant result was the conceptual clarification: the validation set is a "simulated test" during training, while the true test set is completely unseen data. This directly informs my research plan: the SIS+shear dataset will be split for training/validation, and the final model will be evaluated on the completely separate SIE dataset as the true test set.
### Challenges & Pause Points

- I had to pause to fully grasp why `validation_split` taking the last segment of data was problematic. I worked through the example array `[1, 2, 3, ..., 10]` to visualize the bias that would be introduced if the data was ordered.
- The distinction between "validation" and "test" was initially subtle. I had to consult multiple sources (video, documentation) to solidify my understanding that validation is for _model development_ and test is for _model assessment_.
### Questions & Ideas

- For my specific research data (SIS+shear), is it better to use `validation_split` or manually create a `validation_data` set? Would manual creation give me more control, for example, to ensure the validation set is representative of the entire data distribution?
	- When I implement this, I should write a function to plot training accuracy/loss and validation accuracy/loss on the same graph over each epoch. This will make it visually immediate to spot overfitting/underfitting.
- From what I have learned, I know that validation accuracy should be close to training accuracy. What if my validation accuracy is consistently _higher_ than my training accuracy as was seen in the video? Is this indicative of something? Might need to learn more about this. 
### Next Steps

1. Check if Amarel was able to finish producing the mock lens data tomorrow. 
2. Apply the knowledge I gained today by modifying the provided sequential model for my data and draw out a detailed network for the data I have. This will help in determining the input shape and the outputs I want.
3. Try a different model than the given sequential model? Use Keras' website to learn how different models are implemented and research this.
4. Also, follow up on the idea from point number 2 and implement the training/validation accuracy & loss plotting function that Keras provides to double check if visually the model is implemented how I drew it once I get there.
5. Experiment with different validation split sizes (e.g., 0.2, 0.3) to see how it affects model performance and the stability of the validation metrics?
6. After achieving a model that performs well on the validation set, the final step in network creation will be to run it on the held-out SIE and SIE+shear test sets for the final inference results from the model. 
7. Use the network for a real gravitational lens system with known parameters to further check for accuracy in the network's ability to make inferences.
#### Note to Self:

- Ended the ML video at 42:33, continue from there the next time.

***
## Research Log 10-14-2025
### Today's Focus

Optimizing and executing the large-scale batch job by implementing a parallel processing strategy using a SLURM job array. A secondary focus was planning for future data preprocessing and model robustness based on advisor feedback.
### What I was able to accomplish

- Diagnosed the failure of the original single, long-running batch job due to the 48-hour time limit.
- Collaborated with my professor to devise a new strategy: run 10 parallel jobs with a reduced number of iterations (n=10⁴) each, instead of one job with n=10⁵.
- Successfully modified the Python script to use the `SLURM_ARRAY_TASK_ID` for generating unique output files, preventing overwrites.
- Wrote and configured a new SLURM script utilizing `--array=1-10` to launch the 10 parallel jobs.
- Developed and tested a robust Python utility (`combine_results()`) to merge the outputs from all parallel jobs into three final combined arrays.
### Results

- The proof-of-concept with 10 mock lenses was successful.
- The 10-job SLURM array was submitted and ran successfully.
- The file-naming strategy worked perfectly, resulting in 10 sets of three files (e.g., `valShear_1.npy` ... `valShear_10.npy`).
- The `combine_results()` function is ready to execute once all jobs are complete.
### Challenges & Pause Points

- The main batch job hitting the 48-hour wall-clock limit on the cluster.
- Dr. Keeton recommended a "dumb parallel processing" approach to work around the time limit, which was successfully implemented.
- Waiting for the 10 parallel jobs to complete their 48-hour run. The process is currently in progress.
- How to preprocess our data to be invariant to translations and rotations, which is crucial for a robust model?
### Questions & Ideas

- If the jobs were to fail again, then is there a more efficient way to split the work? Should we consider more jobs with even smaller `n`?
- How do Convolutional Neural Networks (CNNs) achieve translation and rotation invariance? What is the core mechanism (e.g., convolutional layers, pooling, data augmentation) and how can we adapt it for our regression task, since we are not building a classifier?
- <u>Idea</u>: Automate the post-processing by having the SLURM script run `combine_results.py` as a dependent job. 
### Next Steps

1. **Short-term:**
    - Verify that all 10 jobs in the SLURM array have completed successfully.
    - Run the `combine_results.py` script to merge the individual output files.
    - Perform a preliminary data quality check on the combined arrays.
2. **Medium-term:**
    - **Remind Dr. Keeton** about the Potential Array for the lensing system.
    - Begin the planned analysis on the combined dataset.
3. **Research & Development:**
    - **Read deeply** into the architectural principles of CNNs, specifically focusing on how convolutional and pooling layers provide translation invariance.
    - Investigate how these principles (e.g., using convolutional layers in a regression network) can be applied to our specific problem to make our model robust to rotations and translations in the input data.

***
## Research Log 10-15-2025 to 10-16-2025
### The Focus for these Two Days

Exploring neural network architectures that maintain performance under translational and rotational transformations of gravitational lensing data. Investigating the mathematical foundations of group theory and its application to equivariant and invariant neural networks for astrophysical data analysis.
### What I was able to accomplish

- Successfully set up and launched 10 parallel batch jobs on Amarel for mock lens generation
- Studied fundamental concepts of group theory and symmetry operations
- Learned about the mathematical distinctions between equivariance,
```math
f(t(x)) = t'(f(x)),
```
  and invariance,
```math
f(t(x)) = f(x).
```
- Explored how these concepts apply to gravitational lensing systems where image positions may be rotated or translated
- Continued investigating sequential neural network implementations in Keras and researched more about how the neural network is saved using `model.save()` and `model.load()`. 
	- I found that this saves the model architecture as well as the weights and biases if the model had been trained prior to saving.
	- What I found interesting here is the models can also be saved architecturally (by saving it as a json) and that the weights can be saved separated as well!
### Results

- Batch jobs running successfully with proper task ID handling and file organization
- Clear understanding that for gravitational lens analysis, invariance is preferred over equivariance for consistent predictions regardless of coordinate transformations (however, maybe equivariance can be used to get invariance such that no transformation of the output takes place)
- Recognition that while equivariance preserves transformation relationships, invariance provides unchanging outputs.
- Conceptual framework for potentially reversing output transformations when the input transformation is known (as mentioned earlier).
### Challenges & Pause Points

- Determining whether to pursue pure invariance or develop transformation-reversal methods for equivariant outputs
- Understanding the complex mathematical foundations of group theory and its practical implementation in neural networks
- Balancing theoretical learning with practical implementation timelines
- Ensuring the neural network architecture will generalize well to real observational data after training on simulations
### Questions & Ideas

- Could we implement a hybrid approach: use equivariant networks but include a transformation reversal layer to achieve effective invariance?
- How do we mathematically characterize the transformation groups relevant to gravitational lensing (rotation, translation groups)?
- Would data augmentation with random rotations/translations during training be sufficient, or do we need architecturally enforced invariance?
- Is there existing work in astrophysics applying group-equivariant networks to similar problems?
- Could we leverage the known physical symmetries of gravitational lensing equations to inform the network architecture or encode the lensing equation as a metric into a tensor? Should I even consider tensors?
### Next Steps

1. Monitor batch job completion and combine results using the prepared script
2. Design initial neural network architecture focusing on transformation invariance
3. Implement basic sequential network in Keras as baseline
4. Research existing equivariant neural network implementations (e.g., group-equivariant CNNs or ENNs further)
5. Develop strategy for testing network performance under coordinate transformations?    
6. Plan data preprocessing pipeline that maintains physical meaningfulness while enabling machine learning

***
## Research Log 10-17-2025

***
## Research Log 10-20-2025

***
## Research Log 10-21-2025

***
## Research Log 10-22-2025 to 10-23-2025

***
## Research Log 10-24-2025

***
## Research Log 10-27-2025

***
## Research Log 10-28-2025

***
## Research Log 10-29-2025 to 10-30-2025

***
## Research Log 10-31-2025

***
## Research Log 11-03-2025

***
## Research Log 11-03-2025

***
## Research Log 11-04-2025

***
## Research Log 11-06-2025

***
## Research Log 11-07-2025 to 11-10-2025

***
## Research Log 11-11-2025

***
## Research Log 11-12-2025

***
## Research Log 11-13-2025

***
## Research Log 11-14-2025

***
## Research Log 11-17-2025

***
## Research Log 11-18-2025 to 11-24-2025

***
## Research Log 11-25-2025

***
## Research Log 11-26-2025

***
## Research Log 11-28-2025

***
## Research Log 12-01-2025

***
## Research Log 12-02-2025

***
## Research Log 12-03-2025

***
## Research Log 12-04-2025

***
## Research Log 12-05-2025

***
## Research Log 12-08-2025

***
## Research Log 12-09-2025

***

