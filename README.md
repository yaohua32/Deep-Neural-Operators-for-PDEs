# Deep-Neural-Operators-for-PDEs
This repository implements several deep neural operator (DNO) frameworks for solving parametric partial differential equations (PDEs) and related inverse problems. These methods aim to learn an operator $\mathcal{G}$ that maps input parameters (e.g., coefficients, boundary/initial conditions) to the corresponding PDE solutions.

#### 🎯 Problem Formulation
Given a parametric PDE $\mathcal{N}[u;a]=0$,
where $\mathcal{N}$ represents the PDE, $a$ denotes the input function (e.g., coefficients or boundary/initial conditions), $u$ is the corresponding PDE solution. The goal of deep neural operators is to approximate the mapping: $\mathcal{G}: a \in \mathcal{A} \rightarrow u \in \mathcal{U}$.

## 🏆 Implemented Deep Neural Operator Frameworks
DNO methods can be categorized into **data-driven** and **physics-aware** approaches based on whether they incorporate physics constraints during training.
### 1. Data-Driven DNOs (Supervised Learning)
These methods learn the operator solely from labeled training pairs $(a,u)$.
- **[DeepONet](https://arxiv.org/abs/1910.03193)**: A pioneering architecture using branch and trunk networks.
- **[Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895)**: Employs Fourier transformations for efficient global convolutions.
- **[MultiONet](https://arxiv.org/abs/2502.06250):**  Enhances DeepONet with shortcut connections, significantly improving approximation power with minimal parameter increase.

### 2. Physics-Aware DNOs (Self-Supervised Learning)
These methods incorporate physics constraints (i.e., PDE residuals) into the training process, reducing data requirements and improving accuracy and generalization.
  - **[PI-DeepONet](https://arxiv.org/abs/2103.10974):** Extends DeepONet by adding PDE residuals as training constraints. However, it requires higher regularity of inputs/outputs, making it unsuitable for singular or discontinuous inputs/outputs.
  - **[PINO](https://arxiv.org/abs/2111.03794):** A physics-informed extension of FNO that leverages PDE residuals but struggles with complex geometries and high-dimensional problems due to its reliance on regular, fine meshes for Fourier transformation and derivative approximation.
  - **[PI-MultiONet](https://arxiv.org/abs/2502.06250):** A physics-informed version of MultiONet, improving accuracy while reducing labeled data requirements. 
  - **[Deep Generative Neural Operator (DGenNO)](https://arxiv.org/abs/2502.06250):** A novel framework leveraging **deep generative modeling** and **probabilistic latent variables** to handle complex physics-based problems, including inverse problems. DGenNO offers several key advantages:
    -  Enable to **learn purely from physics constraints**.
	- Effectively solves parametric PDEs and inverse problems with **discontinuous inputs**.
	- Provides probabilistic estimates and **robust performance with sparse, noisy data** in solving **inverse problems**.
	- Uses **weak-form PDE residuals** based on compactly supported radial basis functions (CSRBFs), reducing regularity constraints.

## 📌 Benchmark Problems
We evaluate the DNO frameworks on the following PDEs:
### 1. Burger’s Equation
Goal: Learn the operator mapping initial condition $a(x):=u(x,t=0)$ to the solution $u(x,t)$.

### 2. Darcy’s Flow
Goal: Learn the mapping from the permeability field $a(x)$ to the pressure field $u(x)$.
We considered two cases: (1) Smooth $a(x)$ and (2) Piecewise-constant $a(x)$.

### 3. Stokes Flow with a Cylindrical Obstacle
Goal: Learn the mapping from in-flow velocity ${\bf u}_0 = (a(x), 0)$ to the pressure field $u(x)$.

### 4. Inverse Discontinuity Coefficient in Darcy’s Flow

We also consider the inverse problem of reconstructing the **piecewise-constant** permeability field $a(x)$ from **sparse, noisy** observations of $u$. This problem has important applications in subsurface modeling and medical imaging.

📌 **Remark**: Due to the challenging nature of this inverse problem, the above DNO frameworks **are unable to solve it except for the Deep Generative Neural Operator (DGenNO) method**. Therefore, we have only implemented DGenNO for this inverse problem.

## 🔗 Data Availability
- **All Physics-aware DNOs** in this repository are trained exclusively using physics information (i.e., **without labeled (a, u) pairs**).
- Training data (only for data-driven DNOs) and testing data can be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1MOFme5DgUd339rlL1IGq35ZcVCR0CWqa?usp=drive_link)**.

## 🔬 Future Work

This repository is an ongoing project, and more DNO frameworks and PDE applications will be added. We welcome contributions and collaborations!

## 📖 Citation
If you find this work useful or are interested in our **DGenNO** or **PI-MultiONet** methods, please cite our paper:
```
@article{zang2025dgno,
  title={DGenNO: A Novel Physics-aware Neural Operator for Solving Forward and Inverse PDE Problems based on Deep, Generative Probabilistic Modeling},
  author={Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={arXiv preprint arXiv:2502.06250},
  year={2025}
}
```
