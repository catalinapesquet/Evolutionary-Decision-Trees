# Scientific Context
This project is inspired by the work of Márcio Basgalupp, Rodrigo Barros, and Vili Podgorelec, "Evolving Decision-Tree Induction Algorithms with a Multi-Objective Hyper-Heuristic" (SAC 2015)​057 Evolving decision-t….
Their research introduced a hyper-heuristic evolutionary framework capable of automatically designing decision-tree induction algorithms, balancing predictive performance and model complexity through multi-objective optimization.

Building on these foundations, our project aims to extend and adapt these ideas to create evolutionary decision-tree induction algorithms that are not specialized for a single dataset, but rather designed to perform well across a group of datasets within the same application domain.

The primary goal is to evolve general-purpose, robust, and comprehensible decision-tree induction algorithms tailored to specific fields of application, while maintaining high predictive performance and controlled model complexity.

# Description
This research project is focused on the implementation and optimization of evolutionary decision trees capable of:

* robustly handling missing values,

* leveraging various advanced split criteria,

* applying different pruning strategies to control complexity,

* being automatically optimized by evolutionary algorithms (NSGA-II),

* evaluated through multi-objective optimization (e.g., Recall, F1-score, model complexity).


# Main features
* Eight different strategies for handling missing values.

* Five advanced split criteria (Gini, Information Gain, Gain Ratio, G-Statistic, etc.).

* Five post-pruning methods (Reduced-Error Pruning, Pessimistic Error Pruning, Minimum Error Pruning, Cost-Complexity Pruning, Error-Based Pruning).

* Automatic generation of trees encoded as genes.

* Multi-objective optimization using NSGA-II (via Pymoo).

* Graphical visualization of evolved trees.


