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


# EMO-AutoML-DT: Evolutionary Multi-Objective AutoML for Decision Trees

This repository implements a hyper-heuristic evolutionary framework for automatically designing decision tree induction algorithms. It builds upon and extends the work of Basgalupp et al. ("Evolving Decision-Tree Induction Algorithms with a Multi-Objective Hyper-Heuristic", SAC 2015), by aiming for **generalization across datasets**, **robustness**, and **model interpretability**.

## Scientific Context

This project builds upon a research line initiated by Rodrigo Barros and colleagues, notably through the development of:

- **HEAD-DT** (Barros et al., Evolutionary Computation, 2013): a hyper-heuristic evolutionary algorithm to design decision tree induction heuristics via single-objective optimization.
- **MOHEAD-DT** (Barros et al., 2015, SAC): an extension of HEAD-DT into the **multi-objective** domain using NSGA-II, optimizing both predictive accuracy and model complexity.

These methods demonstrated that it is possible to automatically evolve full induction algorithms (not just models) by encoding tree-building heuristics as chromosomes and optimizing them via evolutionary computation.

Building on this foundation, **our project goes one step further** by:

- targeting **robust, general-purpose algorithms** that perform well across a *group of datasets* in a shared application domain, not just dataset-specific tuning;
- expanding the search space to include **13 split criteria**, **5 stopping rules**, **8 missing value strategies**, and **5 pruning methods**;
- leveraging **multi-objective optimization** (recall, specificity, model complexity) via NSGA-II to guide the search toward interpretable and performant trees.

This approach enables the discovery of **new combinations of known heuristics**, automatically tuned for optimal trade-offs between performance and simplicity.


Our framework targets **general-purpose, comprehensible, and high-performing algorithms** for classification, evaluated over multiple UCI datasets. The final solutions are selected on the Pareto front balancing:
- **Recall**
- **Specificity**
- **Tree Complexity** (number of nodes)

## Project Highlights

-  **13 split criteria** including Gini, Information Gain, Gain Ratio, G-statistic, CAIR, Twoing, etc.
-  **5 stopping rules**: max depth, homogeneity, predictive accuracy, etc.
-  **8 strategies for handling missing values**, both during training and inference.
-  **5 pruning methods**: REP, PEP, MEP, CCP, and EBP.
-  **Multi-objective optimization** using NSGA-II (via `pymoo`)
-  **One-point crossover and mutation** for evolutionary variation.
-  Benchmarked across 15+ UCI datasets: Iris, Car, Dermatology, Glass, Wine Quality, etc.

## Repository Structure



