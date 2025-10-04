Quasi-Oppositional Differential Evolution (QODE) is an enhanced version of the traditional Differential Evolution (DE) algorithm that accelerates convergence by using quasi-oppositional concepts during initialization and generation phases. The main goal is to explore the search space more efficiently and avoid local optima.

Pseudocode:

Initialize control parameters: population size NP, scaling factor F, crossover rate CR, and maximum generations Gmax.

Generate an initial population P of NP individuals randomly within the defined bounds.

For each individual Xi in P, compute its quasi-opposite individual Xqi using the relation:
Xqi = lower_bound + upper_bound − Xi + random() × (Xi − (lower_bound + upper_bound)/2)

Combine original and quasi-opposite populations, evaluate fitness for all, and select the best NP individuals to form the initial population.

For generation G = 1 to Gmax:
a. For each individual Xi in P:
i. Randomly select three distinct individuals Xa, Xb, and Xc ≠ Xi.
ii. Perform mutation to create a donor vector: Vi = Xa + F × (Xb − Xc).
iii. Apply crossover between Xi and Vi to produce a trial vector Ui:
For each dimension j:
If rand(0,1) < CR or j == jrand: Uij = Vij else Uij = Xij
iv. Evaluate fitness of Ui.
v. If fitness(Ui) < fitness(Xi), replace Xi with Ui.
b. Apply quasi-oppositional generation phase:
For each individual Xi, compute its quasi-opposite Xqi and evaluate its fitness.
If fitness(Xqi) < fitness(Xi), replace Xi with Xqi.
c. Record the best individual of the generation based on fitness.

Repeat steps until maximum generations are reached or convergence criterion is satisfied.

Return the best solution found as the global optimum.

The QODE algorithm improves upon standard DE by integrating quasi-oppositional learning, enhancing diversity, and accelerating convergence speed without compromising global exploration.
