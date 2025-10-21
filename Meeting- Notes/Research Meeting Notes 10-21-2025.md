### Notes

- Time delay surface helps describe the images in terms of saddles and local minima. The images inside Einstein ring are saddle points and the images outside the einstein ring are local minima. The images that are farthest away from the source produce the least time delay and "arrive first" as the gravitational potential is the lowest and the geometric potential is the highest, but the gravitational potential outweighs the geometric potential and thus, we see the image arrive first. Then the images inside the Einstein ring arrive after the outer images have arrived. 
	- Time delay surface is a function that describes time delays as saddles or local minima (fermat's principle).
- For conv networks, will need to convert data into 2D images
- For the current data set, don't necessarily need 2D pixel data as that might waste memory, but can explore this route and create "images" for mock lens data and use that as training data for a GCNN
- Could also help to explore different types of NNs and compare the NNs to each other
- One note to make regarding transformations is that if the Einstein radius was scaled by a factor, say $w$, and the source position was also scaled by that same factor $w$, then the lensed images would closer together or farther apart also by the same factor $w$.
- Create python script to preprocess data and sort by time delays.