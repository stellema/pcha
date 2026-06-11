# Archetypal analysis using Principal Convex Hull Approximation (PCHA) 

Principal Convex Hull Approximation (PCHA) is a matrix factorisation / unsupervised machine learning method that models multi-dimensional as convex combinations of archetypes (i.e., it finds patterns that are extreme). 

## Description

Archetypal analysis (AA) was introduced by Cutler and Breiman (1994) as a data clustering method used to identify and differentiate between extreme configurations in point sets. PCHA finds archetypes, and the combination of archetypes that the data represents at each time step.

The PCHA scripts are derived from the original MATLAB implementation described in Mørup and Hansen (2012). Both the R and Python implementations follow the same overall workflow, typically consisting of three stages.

1. Data loading and preparation
    The first stage constructs a NaN‑free data matrix

    $X = X[\text{space}, \text{time}]$ or $X = X[\text{features}, \text{observations}]$,

    which serves as the input to the PCHA workflow. The algorithm solves

        $Argmin_{C, S} ​∥X − XCS∥^2_{Frobenius}$​,

    where $C$ and $S$ are stochastic matrices, such that

        $XCS=X[\text{space}, \text{time}]⋅C[\text{time}, \text{cardinality}]⋅S[\text{cardinality}, \text{time}]$,

    for a prescribed cardinality (i.e., number of clusters or classes), set a priori.

    This representation of $X$ is a form of matrix factorisation, analogous to the well‑known singular value decomposition $X = USVT$.

2. Optimisation (PCHA engine)

    The main PCHA engine is wrapped in a routine called `PCHA_multi_init`, as the method typically requires multiple restarts to avoid convergence to local minima of the objective function above. Several initialisation strategies are available; if you have suggestions for improved approaches, please let me know.

    At this stage, careful tuning of the number of iterations and the (relative) convergence criterion is important to ensure proper convergence. Enabling the `verbose_inner` option provides additional diagnostic information during optimisation.

3. Diagnostics and visualisation

    The final stage illustrates the resulting $C$ and $S$ matrices, along with the associated archetypal patterns.

    For more information, please see the list of papers included in the references that use variations of this code and method.

## Getting Started

* Currently, this package must be downloaded and run locally.
* Please see `worked_example_sst.ipynb` for an example of how to use pcha

## License

This project is licensed under the Creative Commons License - see the LICENSE file for details.

## Acknowledgments

Several people have contributed to the development and modification of this code over time, including Didier Monselesan, Ulf Aslak, Dino Jericevic, Jieyang Hu and Annette Stellema.

This code was adapted from the package developed by Mørup and Hansen (2012) and Ulf Aslak (https://github.com/ulfaslak/py_pcha.git). Didier Monselesan developed and added the global optimison wrapper, created the worked example and converted the original MATLAB implementation to Python with help of generative AI tools (Google AI and Microsoft Copilot). Annette Stellema created and maintains this repository.

## References

Black, A. S., Monselesan, D. P., Risbey, J. S., Sloyan, B. M., Chapman, C. C., Hannachi, A., Richardson, D., Squire, D. T., Tozer, C. R., & Trendafilov, N. (2022). Archetypal analysis of geophysical data illustrated by sea surface temperature. Artificial Intelligence for the Earth Systems, 1(3), Article e210007. https://doi.org/10.1175/AIES-D-21-0007.1


Chapman, C. C., Monselesan, D. P., Risbey, J. S., & Feng, M., & Sloyan, B. M. (2022). A large-scale view of marine heatwaves revealed by archetype analysis. Nature Communications, 13(1), 7843.


Chapman, C. C., Monselesan, D. P., Risbey, J. S., Hannachi, A., Lucarini, V., & Matear, R. (2025). The typicality of regimes associated with Northern Hemisphere heatwaves. Journal of Climate, 38(15), 3729–3750. https://doi.org/10.1175/JCLI-D-24-0548.1


Cutler, A., & Breiman, L. (1994). Archetypal analysis. Technometrics, 36(4), 338–347. https://doi.org/10.2307/1269949


Hannachi, A., Finke, K., Trendafilov, N., Monselesan, D., Risbey, J., Chapman, C., & Chafik, L. (2026). Weather and climate extremes: Simplex, dynamical systems and hull clustering. Journal of Geophysical Research: Atmospheres, 131, e2025JD045044. https://doi.org/10.1029/2025JD045044


Monselesan, D. P., Risbey, J. S., Legrésy, B., Cravatte, S., Pagli, B., Izumo, T., Chapman, C. C., Freund, M., Hannachi, A., Irving, D., et al. (2024). On the archetypal “flavours,” indices and teleconnections of ENSO revealed by global sea surface temperatures. arXiv. https://arxiv.org/abs/2406.08694


Mørup, M., & Hansen, L. K. (2012). Archetypal analysis for machine learning and data mining. Neurocomputing, 80, 54–63. https://doi.org/10.1016/j.neucom.2011.06.033


Richardson, D., Black, A. S., Monselesan, D. P., Moore, T. S., Risbey, J. S., Schepen, A., Squire, D. T., & Tozer, C. R. (2021). Identifying periods of forecast model confidence for improved subseasonal prediction of precipitation. Journal of Hydrometeorology, 22(2), 371–385.


Risbey, J. S., Monselesan, D. P., Black, A. S., Moore, T. S., Richardson, D., Squire, D. T., & Tozer, C. R. (2021). The identification of long-lived Southern Hemisphere flow events using archetypes and principal components. Monthly Weather Review, 149(6), 1987–2010. https://doi.org/10.1175/MWR-D-20-0314.1


Risbey, J. S., Monselesan, D. P., Chapman, C. C., Chung, C., Hannachi, A., Irving, D., Parker, T., Pook, M. J., Ramesh, N., Stellema, A., Tozer, C. R., & Kala, J. (2025). Extreme monthly rainfall archetypes for Australia. Journal of Southern Hemisphere Earth Systems Science, 75(3). https://doi.org/10.1071/ES25016