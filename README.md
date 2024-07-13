# Project - Optimization of Mandelbrot set computation

---

90535 - High Performance Computing

Laurea Magistrale - Computer science

Artificial Intelligence Track

University of Genova

---

Student: Arnaud Ruymaekers

---

In this project, I attempted optimize the computation of the Mandelbrot set. First a profiling has been performed to find out the hotspot of the computation, then the different parallelization and vectorization techniques learned were applied. These techniques were implemented with OpenMP, MPI, a hybrid of both, and finally CUDA. Within these implementations, different parameterizations were tried to find out the most optimal. Finally, all the runtimes were compared among each other to see if the theory matches to practice and to determine which was the most optimal implementation.