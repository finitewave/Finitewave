---
title: 'Finitewave: a lightweight and accessible framework for cardiac electrophysiology simulations'
tags:
  - Python
  - cardiac modeling
  - simulations
  - electrophysiology
  - reentries
  - fibrosis
authors:
  - name: Timur Nezlobinsky
    orcid: 0009-0005-8299-5295
    affiliation: 1
    corresponding: true
  - name: Arstanbek Okenov
    orcid: 0009-0008-8435-9921
    affiliation: 1
  - name: Nele Vandersickel
    orcid: 0000-0002-8827-5302
    affiliation: 1
  - name: Alexander V Panfilov
    orcid: 0000-0003-2643-642X
    affiliation: 1
affiliations:
 - name: Ghent University, Belgium
   index: 1
date: 5 August 2025
bibliography: paper.bib

---

# Summary
The growing progress in cardiac and medical modeling opens broad opportunities for the development of tools and software packages that enable simulations
for both research and clinical purposes. Mathematical modeling and numerical simulation have become essential for understanding cardiac arrhythmias, testing hypotheses in silico, and supporting the development of medical devices and therapies [@Jaffery_2024; @Trayanova_2024].

We present **Finitewave**, a lightweight and extensible Python framework designed for fast, reproducible, and accessible simulations of cardiac electrical propagation using finite difference methods. Finitewave significantly lowers the entry barrier for students, researchers, and users without
a strong technical background. Its modular structure, fast numerical core, and seamless integration with Python's scientific ecosystem allow for both rapid prototyping and advanced analysis using standard tools for visualization and post-processing.


# Statement of need
Over the past decades, numerous tools have been developed to simulate normal and pathological cardiac activity, including electrical propagation, contraction, and anatomical features [@Plank_openCARP; @Cooper_Chaste; @africa2022lifexcore; @Clerx2016Myokit; @Finsberg2023; @KABUS2026110088]. As cardiac models have become more detailed and biophysically accurate, their computational cost and complexity have increased. Some existing frameworks provide powerful capabilities for large-scale and physiologically detailed simulations, but are often designed with high-performance computing workflows in mind and may require substantial setup effort and domain-specific expertise, which can limit accessibility for users without a strong computational background. Other tools provide a more lightweight and user-friendly environment, but tend to focus on reduced-complexity or single-cell workflows.

**Finitewave** addresses this gap by offering a lightweight, transparent, and Python-native framework for cardiac modeling. It is designed to enable early user engagement in the modeling process, with a clear and modular structure that supports experimentation, learning, and customization.
Its Python foundation allows smooth integration with other libraries (e.g., NumPy, Matplotlib, SciPy, Jupyter) and makes it ideal for use in educational and research settings.

Finitewave is particularly suited for:

- **Research**: Studying wave propagation, reentry, or arrhythmias under various physiological conditions (e.g., fibrosis).

- **Hypothesis testing**: Rapid prototyping of ideas and simulation protocols.

- **Education**: Teaching modeling concepts through its modular and readable design.

- **Custom development**: Creating tailored solutions via native Python integration.

- **Dataset generation**: Producing synthetic data for machine learning or statistical analysis.

This positions Finitewave as a complementary and accessible alternative to existing platforms, offering a more flexible and user-friendly environment for cardiac modeling and experimentation.

# Usage philosophy

Finitewave supports both 2D and 3D simulations, offering an open and interactive space for implementing a wide range of computational experiments.
A minimal working script requires only a few lines of clearly structured code, making it easy to get started. For exploratory use, simulations can also be run inside Jupyter notebooks, enabling immediate visualization and interactive analysis.

Advanced users can easily extend base scripts with custom metrics, animations, or protocol logic. This makes it possible to scale from simple demonstrations to complex simulations while retaining full transparency and control over each step.

The repository includes detailed examples, covering the main features of the framework, as well as tutorials demonstrating how Finitewave can be used  for different types of research tasks. Our goal is to make Finitewave not only convenient for experienced users but also an accessible and modern entry point into cardiac modeling for students and early-career researchers.

Despite being a relatively new open-source project, Finitewave has already been used as the primary simulation tool in at least two peer-reviewed publications [@Nezlobinsky_multiparam; @Okenov_comp_based].

## Acknowledgements

The authors thank Bjorn Verstraeten, Sebastiaan Lootens, Viktor Van Nieuwenhuize, and Robin Van Den Abeele for helpful discussions, testing, and feedback during the development of Finitewave.

# References
