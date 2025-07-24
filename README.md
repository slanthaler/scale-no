The code for Scale-consistency learning for partial differential equations.

 Machine learning (ML) models have emerged as a promising approach for solving
partial differential equations (PDEs) in science and engineering. Previous ML
models typically cannot generalize outside the training data; for example, a
trained ML model for the Navier-Stokes equations only works for a fixed
Reynolds number ($Re$) on a pre-defined domain. To overcome these limitations,
we propose a data augmentation scheme based on scale-consistency properties of
PDEs and design a scale-informed neural operator that can model a wide range of
scales. Our formulation leverages the facts: (i) PDEs can be rescaled, or more
concretely, a given domain can be re-scaled to unit size, and the parameters
and the boundary conditions of the PDE can be appropriately adjusted to
represent the original solution, and (ii) the solution operators on a given
domain are consistent on the sub-domains. We leverage these facts to create a
scale-consistency loss that encourages matching the solutions evaluated on a
given domain and the solution obtained on its sub-domain from the rescaled PDE.
Since neural operators can fit to multiple scales and resolutions, they are the
natural choice for incorporating scale-consistency loss during training of
neural PDE solvers. We experiment with scale-consistency loss and the
scale-informed neural operator model on the Burgers' equation, Darcy Flow,
Helmholtz equation, and Navier-Stokes equations. With scale-consistency, the
model trained on $Re$ of 1000 can generalize to $Re$ ranging from 250 to 10000,
and reduces the error by 34% on average of all datasets compared to baselines.

