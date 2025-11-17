### Today's Focus

Setting up the PyTorch G-CNN implementation and troubleshooting extensive compatibility issues with modern Python and CUDA toolchains. The goal was to get a basic working instance that could run the rotated MNIST benchmark as proof of concept.
***
### What I was able to accomplish

- Successfully cloned the PyTorch G-CNN repository and identified its specific requirements (CUDA 11.3, PyTorch 1.12.1).
    
- Installed CUDA 11.3 alongside existing CUDA 13.0 installation, managing multiple CUDA versions through environment variables.
    
- Attempted to build C++ and CUDA extensions for performance optimization, encountering multiple compilation errors due to version mismatches.
    
- Created systematic debugging approach by testing individual components (data loading, model definition, training loop) separately to isolate issues.
***
### Results

- C++ extensions compiled successfully but CUDA extensions failed due to PyTorch/CUDA version incompatibilities.
    
- Discovered that the required PyTorch 1.12.1 + CUDA 11.3 combination is no longer available from standard repositories, forcing alternative approaches.
    
- Verified that basic data loading and model definition work without optimized extensions, though with significantly reduced performance.
    
- Confirmed the repository structure and identified key components (GConv2d, GMaxPool2d layers) that would need modification for lensing applications.
***
### Challenges & Pause Points

- Modern PyTorch versions incompatible with repository's CUDA 11.3 requirement, and old PyTorch versions no longer available.
    
- CUDA extension compilation failures due to API changes between CUDA versions and missing dependencies.
    
- Complex build process with multiple setup files (setup_cpp.py, setup_cuda.py) requiring careful environment configuration.
    
- The repository appears to be a research implementation rather than production-ready code, with limited documentation and error handling.
***
### Questions & Ideas

- Is the performance gain from CUDA extensions essential for initial experiments, or can we proceed with slower pure-PyTorch implementations?
    
- Would it be more efficient to reimplement the core G-CNN layers in modern PyTorch rather than debugging this specific repository?
    
- How critical are the optimized C++/CUDA kernels for the gravitational lensing application scale? Our datasets may be small enough that Python overhead is acceptable initially.
***
### Next Steps

1. Attempt to patch the code to work with current PyTorch versions while maintaining G-CNN functionality.
    
2. If patching fails, consider implementing a minimal G-CNN from scratch based on the paper's methodology.
    
3. Test the pure-PyTorch fallback implementation to verify mathematical correctness even without performance optimizations.
    
4. Begin planning the adaptation from image classification (MNIST) to parameter regression (lensing parameters).
##### Tags: #Patching #FixingErrors #NeuralNetwork #GroupTheory #GCNN 




