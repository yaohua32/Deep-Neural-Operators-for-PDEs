# Deep-Neural-Operators-for-PDEs
This repository implements several deep neural operator (DNO) frameworks for solving parametric partial differential equations (PDEs) and related inverse problems. These methods aim to learn an operator $\mathcal{G}$ that maps input parameters (e.g., coefficients, boundary/initial conditions) to the corresponding PDE solutions.

#### 🎯 Problem Formulation
Given a parametric PDE:
$$
\begin{cases}
\mathcal{N}[u; a] &= 0 \\
\mathcal{B}[u; a] & =0
\end{cases} \tag{1}
$$
where $\mathcal{N}$ represents the PDE, $\mathcal{B}$ represents the boundary/initial conditions, $a$ denotes the input function (e.g., coefficients or boundary/initial conditions), $u$ is the corresponding PDE solution. The goal of deep neural operators is to approximate the mapping:
$$
\mathcal{G}: a \in \mathcal{A} \rightarrow u \in \mathcal{U}. \tag{2}
$$

## 🏆 Implemented Deep Neural Operator Frameworks
DNO methods can be categorized into **data-driven** and **physics-aware** approaches based on whether they incorporate physics constraints during training.
### 1. Data-Driven DNOs (Supervised Learning)
These methods learn the operator solely from labeled training pairs $(a,u)$.
- **[DeepONet](https://arxiv.org/abs/1910.03193)**: A pioneering architecture using branch and trunk networks.
- **[Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895)**: Employs Fourier transformations for efficient global convolutions.
- **[MultiONet](https://arxiv.org/abs/2502.06250):**  Enhances DeepONet with shortcut connections, significantly improving approximation power with minimal parameter increase.

### 2. Physics-Aware DNOs (Self-Supervised Learning)
These methods incorporate physics constraints (i.e., PDE residuals) into the training process, reducing data requirements and improving accuracy and generalization.
  - **[PI-DeepONet](https://arxiv.org/abs/2103.10974):** Extends DeepONet by adding PDE residuals as training constraints. However, it requires computing high-order derivatives, making it unsuitable for singular or discontinuous functions.
  - **[PINO](https://arxiv.org/abs/2111.03794):** A physics-informed extension of FNO that leverages PDE residuals but struggles with complex geometries and high-dimensional problems due to its reliance on regular, fine meshs for Fourier transformations.
  - **[PI-MultiONet](https://arxiv.org/abs/2502.06250):** A physics-informed version of MultiONet, improving accuracy while reducing labeled data requirements.
  - **[Deep Generative Neural Operator (DGNO)](https://arxiv.org/abs/2502.06250):** A novel framework leveraging **deep generative modeling** and **probabilistic latent variables** to handle complex physics-based problems, including inverse problems. DGNO offers several key advantages:
    -  Enable to **learn purely from physics constraints**.
	- Effectively solves parametric PDEs and inverse problems with **discontinuous inputs**.
	- Provides probabilistic estimates and **robust performance with sparse, noisy data** in solving inverse problems.
	- Uses **weak-form PDE residuals** based on compactly supported radial basis functions (CSRBFs), reducing regularity constraints.

## 📌 Benchmark Problems
We evaluate the DNO frameworks on the following PDEs:
### 1. Burger’s Equation
$$
\begin{split}
u_t + uu_x &= \nu u_{xx}, \quad x \in (0,1),\ t \in(0,1] \\
u(x-\pi, t) &= u(x+\pi, t),\quad t \in(0,1] \\
u(x,t=0) &= a(x)
\end{split} \tag{3}
$$
Goal: Learn the operator mapping initial condition $a(x)$ to the solution $u(x,t)$.

### 2. Darcy’s Flow
$$
\begin{split}
- \nabla( a \nabla u) &= f, \quad x \in \Omega = [0,1]^2 \\
u &= 0
\end{split} \tag{4}
$$
Goal: Learn the mapping from permeability field $a(x)$ to the pressure field $u(x)$.
We considered two cases: (1) Smooth $a(x)$ and (2) Piecewise-constant $a(x)$.

### 3. Stokes Flow with a Cylindrical Obstacle
$$
\begin{split}
-\mu\nabla^2{\bf u} + \nabla p &= {\bf f}, \quad x \in \Omega \\
\nabla\cdot {\bf u} &= 0,\quad x \in \Omega
\end{split} \tag{5}
$$
where ${\bf u}$ is velocity, $p$ is pressure, and $\mu$ is viscosity. Goal: Learn the mapping from in-flow velocity ${\bf u}_0 = (a(x), 0)$ to the pressure field $u(x)$.

### 4. Inverse Coefficient Problem in Darcy’s Flow

We also consider the inverse problem of reconstructing the **piecewise-constant** permeability field $a(x)$ from **sparse, noisy** obserations of $u$. This problem has important applications in subsurface modeling and medical imaging.

## 🔬 Future Work

This repository is an ongoing project, and more DNO frameworks and PDE applications will be added. We welcome contributions and collaborations!

## 📖 Citation
If you find this work useful or are interested in our **DGNO** or **PI-MultiONet** methods, please cite our paper:
```
@article{zang2025dgno,
  title={DGNO: A Novel Physics-aware Neural Operator for Solving Forward and Inverse PDE Problems based on Deep, Generative Probabilistic Modeling},
  author={Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={arXiv preprint arXiv:2502.06250},
  year={2025}
}
```