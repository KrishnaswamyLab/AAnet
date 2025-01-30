## Example notebooks

`AAnet_MNIST_example.ipynb` provides example code for running AAnet on MNIST digit 4, revealing three different archeytpal "four" images in the MNIST database.

`AAnet_simplex_example.ipynb` provides example code for generating simulated nonlinear simplex with four vertices (tetrahedron) and running AAnet, showing AAnet closely identifies the true vertices despite nonlinearity.

`Other_comparisons_simplex_example.ipynb` provides example code for generating simulated nonlinear simplex with four vertices (tetrahedron) and running PCHA, kPCHA, Javadi et al. and Chen et al., demonstrating that other approaches are not able to identify the ground truth vertices.

`AAnet_single-cell_example.ipynb` provides a guide for running AAnet on high-dimensional and noisy single-cell data, including code for learning a low-dimensional and denoised representation with MAGIC and PCA, density subsampling to handle density differences, and decoding archetypes back to the gene expression space.