# Data-driven robust Common Spatial Pattern

A nonlinear eigenvector algorithm for solving data-driven robust Common Spatial Pattern (CSP)

## Common Spatial Pattern (CSP)

Common Spatial Pattern (CSP) [1, 2] is a popular signal-processing technique for feature extraction of Electroencephalography (EEG) signals. It is a data-driven supervised machine learning algorithm that computes spatial filters, which extracts the distinguishing features of the EEG signals associated with a task at hand (such as a motor imagery task).

### Data acquisition and preprocessing

Preprocessing steps (such as channel selection, bandpass filtering, time interval selection) of the raw EEG signals are cruical to successful performance of CSP. Let $Y_c^{(i)}$ denote the preprocessed $i$ th trial for the motor imagery condition $c$, where $c\in \\{-,+\\}$ represents the condition of interest (e.g., left hand and right hand motor imagery). Each trial $Y_c^{(i)}$ is a $n\times t$ matrix, where $n$ is the number of electrode channels and $t$ is the number of sampled time points. Supposing further that the preprocessed trials $Y_c^{(i)}$ are each scaled and centered, the data covariance matrix for a trial is

$$\Sigma_c^{(i)}=Y_c^{(i)}{Y_c^{(i)}}^T \in \mathbb{R}^{n\times n}.$$

Each covariance matrix $\Sigma_c^{(i)}$ is positive definite since $t$ is typically much larger than $n$. Denoting $N_c$ as the number of trials for condition $c$, the average covariance matrix is

$$\overline{\Sigma}\_c=\frac{1}{N_c}\sum\_{i=1}^{N_c}Y_c^{(i)}{Y_c^{(i)}}^T, \qquad \mbox{for }c\in\\{+,-\\}.$$

### Principles of CSP

The goal of CSP is to find filters such that the variance of the spatially filtered signals under one condition is minimized, while that of the other condition is maximized. Specifically, denoting spatial filters as $X\in\mathbb{R}^{n\times n}$ (each column corresponding to a spatial filter), the covariance matrix of the filtered signals in class is given by

$$X^TY_c^{(i)}{Y_c^{(i)}}^T X=X^T\Sigma_c^{(i)}X.$$

Collectively, the covariance matrix of all the spatially filtered signals $Y_c^{(i)}$ in class $c$ is taken as 

$$X^T\overline{\Sigma}_cX.$$

CSP imposes the following couple of constraints on the spatial filters:

1. $X^T\overline{\Sigma}\_{-}X = \Lambda\_{-}$ and $X^T\overline{\Sigma}\_{+}X = \Lambda\_{+}$ such that $\Lambda\_{-}$ and $\Lambda\_{+}$ are diagonal matrices.

2. $X^T(\overline{\Sigma}\_{-} + \overline{\Sigma}\_{+})X=\Lambda\_{-}+\Lambda\_{+}=I_n$ where $I_n$ denotes the $n \times n$ identity matrix.

### Equivalence to a generalized eigenvalue problem and a generalized Rayleigh quotient optimization

These two constraints imply that the spatial filters correspond to the eigenvectors of the generalized eigenvalue problem

$$\overline{\Sigma}\_-x = \lambda (\overline{\Sigma}\_-+\overline{\Sigma}\_+)x.$$

Typically, CSP computes only a subset of the filters that best distinguish the variances between the two classes. Such spatial filters are the extreme eigenvectors of the generalized eigenvalue problem. Let $x_-$ and $x_+$ be the extreme eigenvectors, i.e., $x_-$ is the eigenvector corresponding to the smallest eigenvector and $x_+$ is the eigenvector corresponding to the largest eigenvector. The computation of these extreme eigenvectors $x_-$ and $x_+$ is equivalent to the following generalized Rayleigh quotient

$$\min_{x\neq0} \frac{x^T\overline{\Sigma}\_-x}{x^T(\overline{\Sigma}\_-+\overline{\Sigma}\_+)x}, \qquad \min\_{x\neq0} \frac{x^T\overline{\Sigma}\_+x}{x^T(\overline{\Sigma}\_-+\overline{\Sigma}\_+)x},$$

respectively.

## Robust Common Spatial Pattern

The use of average covariance matrices $\overline{\Sigma}_c$ as estimates of the true covariance matrices poses a major issue due to several factors, such as non-task-related fluctuations, nonstationarity of the EEG signals, and presence of artifacts in the data. As a result, CSP is highly sensitive to noise and prone to overfitting [3, 4].

### Min-max CSP

Instead of limiting the covariance matrices to the fixed covariance, min-max CSP considers covariance matrices in the tolerance sets $\mathcal{S}_c$ of possible covariance matrices. Specifically, the tolerance sets are defined as the ellipsoids of positive definite matrices centered around $\overline{\Sigma}_c$:

$$\mathcal{S}_c=\\{\Sigma_c\mid \Sigma_c\succ0,\\|\Sigma_c-\overline{\Sigma}_c\\|\leq\delta_c\\}$$

where $\delta_c$ denotes the radius of $\mathcal{S}_c$. The robust spatial filters are found by considering the worst-case generalized Rayleigh quotient within the tolerance region $\mathcal{S}_c$. This leads to the min-max optimizations [5]

$$\min_{x\neq0}
    \max_{\substack{\Sigma\_-\in\mathcal{S}\_-\\
        \Sigma_+\in\mathcal{S}\_+}}  
    \frac{x^T\Sigma\_-x}{x^T(\Sigma\_-+\Sigma\_+)x}, \qquad \min\_{x\neq0}
    \max_{\substack{\Sigma\_-\in\mathcal{S}\_-\\
        \Sigma\_+\in\mathcal{S}\_+}}  
    \frac{x^T\Sigma\_+x}{x^T(\Sigma\_-+\Sigma\_+)x}$$

### Data-driven CSP

A data-driven approach is used to construct the tolerance sets [5], where the norm is defined by a PCA-based approach on the data covaraince matrices.

$$\mathcal{S}\_{c } = \bigg\\{ 
\Sigma\_{c} = \overline\Sigma\_{c} +\sum\_{i=1}^m\alpha\_{c}^{(i)} V_{c}^{(i)}\bigg|
\Sigma\_c\succ0,\quad
%\sqrt{\sum\_{i=1}^m \frac{\big(\alpha\_{c}^{(i)} \big)^2}{w_{c}^{(i)}}} = 
\sqrt{\sum\_{i=1}^m\frac{(\alpha\_c^{(i)})^2}{w_c^{(i)}}}
\leq \delta_{c},\quad
\alpha\_{c}^{(i)}\in\mathbb{R}
\bigg\\}.$$

The parameters $V_c^{(i)}$ and $w_c^{(i)}$ are computed by PCA on the data covariance matrices as follows:

1. Vectorize each data covariance matrix $\Sigma_c^{(i)}$ by stacking its columns into a $n^2$-dimensional vector.
2. Compute the covariance matrix $\Gamma_c$ of the vectorized covariance matrices.
3. Compute the $m$ largest eigenvalues and corresponding eigenvectors (principal components) of $\Gamma_c$ as $\\{w\_{c}^{(i)}\\}\_{i=1}^m$ and $\\{V\_{c}^{(i)}\\}\_{i=1}^m$, respectively.
4. Transform the eigenvectors $\\{V_c^{(i)}\\}_{i=1}^m$ into $n\times n$ matrices, symmetrizing afterwards.

### CSP-NRQ

The data-driven min-max CSP becomes a nonliner Rayleigh quotient (NRQ) opimization

$$\min_{x\neq0}\frac{x^T\Sigma_-(x)x}{x^T(\Sigma_-(x)+\Sigma_+(x))x}$$

where for $c\in\\{+,-\\}$,

$$\Sigma_{c}(x) = \overline
\Sigma_{c}+\sum_{i=1}^m\alpha_{c}^{(i)}(x) V_{c}^{(i)}
\quad\text{with}\quad \alpha_{c}^{(i)}(x) = \frac{-c\delta_{c}w_{c}^{(i)}\cdot x^T V_{c}^{(i)}x}{ \sqrt{ \sum\_{i=1}^m w\_{c}^{(i)} \cdot\big(x^TV\_{c}^{(i)}x\big)^2} }.$$

The other min-max CSP follows similarly.

### Linear Clasifier

With the spatial filters $x_-$ and $x_+$ computed, a common classification approach is to use a linear classifier on the log-variance features of the filtered signals. For a given signal $Y$, the log-variance features $f(Y) \in \mathbb{R}^{2}$ are defined as

$$f(Y) =
    \begin{bmatrix}
        \log(x_-^TYY^Tx_-) \\
        \log(x_+^TYY^Tx_+) \\ 
    \end{bmatrix}$$
    
and a linear classifier is defined as

$$\varphi(Y)=w^Tf(Y)-c$$

where the sign of the classifier $\varphi(Y)$ determines the class label of $Y$. The linear weights $w\in\mathbb{R}^{2}$ and $c\in\mathbb{R}$ are determined from the training trials $\{Y_c^{(i)}\}_{i=1}^{N_c}$ using Fisher's linear discriminant analysis (LDA).

## Algorithms

### Fixed-point iteration

One natural idea for solving the CSP-NRQ is to use the fixed-point iteration scheme:

$$x\_{k+1}\longleftarrow\mbox{argmin}\_{x\neq0}\frac{x^T\Sigma\_-(x_k)x}{x^T(\Sigma\_-(x_k)+\Sigma\_+(x_k))x},$$

which is equivalent to the self-consistent field (SCF) iteration 

$$x_{k+1}\longleftarrow\mbox{an eigenvector of }\lambda_{\min}(\Sigma_-(x_k),\Sigma_-(x_k)+\Sigma_+(x_k))$$

for solving a nonlinear eigenvalue problem (NEPv)

$$\Sigma_-(x)x=\lambda_{\min}(\Sigma_-(x)+\Sigma_+(x))x.$$

A major issue with this approach is that the solution of CSP-NRQ does not satisfy the above NEPv.

### Correct NEPv and the corresponding SCF

Let $G_c(x) := \Sigma_c(x) + \widetilde{\Sigma}_c(x)$ where $\widetilde{\Sigma}_c(x)= \sum\_{i=1}^m \nabla\alpha\_{c}^{(i)}(x)x^TV\_{c}^{(i)}$.

If $x$ is a local minimizer of CSP-NRQ, then it is an eigenvector of the NEPv

$$G_-(x)x=\lambda(G_-(x)+G_+(x))x$$

corresponding to the smallest positive eigenvalue [6].

The SCF iteration for solving the correct NEPv is then

$$ x_{k+1}\longleftarrow\mbox{an eigenvector of }\lambda(G_-(x_k),G_-(x_k)+G_+(x_k))$$

where $\lambda$ is the smallest positive eigenvalue. 

## Results

### Convergence Analysis

A synthetic data is generated using the signals created by the following linear mixing model:

$$x(t)=A\begin{bmatrix}s^d(t); s^n(t)\end{bmatrix}+\epsilon(t),$$

where $A$ is a random rotation matrix, $s^d(t)$ represents the discriminative sources, $s^n(t)$ represents the nondiscriminative sources, and $\epsilon(t)$ represents the nonstationary noise.

For the convergence analysis, the nondiscriminative sources $s^n(t)\in\mathbb{R}^8$ are sampled from the standard Gaussian distribution $\mathcal{N}(0,1)$ for both conditions, and the discriminative sources $s^d(t)\in\mathbb{R}^2$ are sampled from $\mathcal{N}(0,\mbox{diag}(0.2,1.4))$ for condition '-' and from $\mathcal{N}(0,\mbox{diag}(1.8,0.6))$ for condition '+'. The nonstationary noise $\epsilon(t)$ is sampled from $\mathcal{N}(0,2)$, regardless of the condition. 50 trials for each condition are created, where each trial $Y_c$ consists of 200 time samples.

<img src="https://user-images.githubusercontent.com/91911643/226500451-340f04d1-1cb8-45d0-9bfc-5cc0f37ae86e.png" width="500" height="350">
<img src="https://user-images.githubusercontent.com/91911643/226500453-34d03e57-2d3d-4663-82ca-d592c86cc260.png" width="500" height="350">

The convergence plots (objective values and errors) show that the Fixed-point iteration converges whereas the SCF iteration for the correct NEPv (Alg. 1) converges rapidly, displaying a local quadratic convergence. 

### Classification Results

The Berlin dataset (available at https://depositonce.tu-berlin.de/items/1b603748-34fe-411c-8fd2-1711925e4101) and the Gwangju dataset (available at http://gigadb.org/dataset/100295) are used to illustrate a classification rate improvement in using robust spatial filters to standard spatial filerts. 

(more classification results using different datasets are avilable in this repo)

Berlin

![CSP_Berlin](https://user-images.githubusercontent.com/91911643/226786267-fab3cbb0-7d53-4934-aecf-c184c43d4c53.png)

Gwangju

![CSP_GWANGJU](https://user-images.githubusercontent.com/91911643/226786296-cdc15019-96bf-4276-b5bc-487ef47aebe0.png)

In the above scatter plots, a dot represents a subject participating in the experiment. A dot above the red diagonal line implies that the corresponding subject obtained improved classification rate from using robust spatial filters. Specifically, 87.5% and 98.0% percent of Berlin and Gwangju subjects, respectively, had improved classificaiton rate.


# References

[1] Zoltan Joseph Koles. The quantitative extraction and topographic mapping of the abnormal components in the clinical eeg. Electroencephalography and clinical Neurophysiology, 79(6):440–447, 1991.

[2] Benjamin Blankertz, Ryota Tomioka, Steven Lemm, Motoaki Kawanabe, and Klaus-Robert Muller. Optimizing spatial filters for robust eeg single-trial analysis. IEEE Signal processing magazine, 25(1):41–56, 2007.

[3] Boris Reuderink and Mannes Poel. Robustness of the common spatial patterns algorithm in the bci-pipeline. University of Twente, Tech. Rep, 2008.

[4] Xinyi Yong, Rabab K. Ward, and Gary E. Birch. Robust common spatial patterns for eeg signal preprocessing. In 2008 30th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, pages 2087–2090. IEEE, 2008.

[5] Motoaki Kawanabe, Wojciech Samek, Klaus-Robert M ̈uller, and Carmen Vidaurre. Robust common spatial filters with a maxmin approach. Neural computation, 26(2):349–376, 2014.

[6] Zhaojun Bai, Ding Lu, and Bart Vandereycken. Robust rayleigh quotient minimization and nonlinear eigenvalue problems. SIAM Journal on Scientific Computing, 40(5):A3495–A3522, 2018.
