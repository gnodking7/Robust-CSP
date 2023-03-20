# Data-driven robust Common Spatial Pattern

A nonlinear eigenvector algorithm for solving data-driven robust Common Spatial Pattern (CSP)

## Common Spatial Pattern (CSP)

Common Spatial Pattern (CSP) \cite{fukunaga2013introduction,blankertz2007optimizing} is a popular signal-processing technique for feature extraction of Electroencephalography (EEG) signals. It is a data-driven supervised machine learning algorithm that computes spatial filters, which extracts the distinguishing features of the EEG signals associated with a task at hand (such as a motor imagery task).

### Data acquisition and preprocessing

Preprocessing steps (such as channel selection, bandpass filtering, time interval selection) of the raw EEG signals are cruical to successful performance of CSP. Let $Y_c^{(i)}$ denote the preprocessed $i$th trial for the motor imagery condition $c$, where $c\in\{-,+\}$ represents the condition of interest (e.g., left hand and right hand motor imagery). Each trial $Y_c^{(i)}$ is a $n\times t$ matrix, where $n$ is the number of electrode channels and $t$ is the number of sampled time points. Supposing further that the preprocessed trials $Y_c^{(i)}$ are each scaled and centered, the data covariance matrix for a trial is

$$\Sigma_c^{(i)}=Y_c^{(i)}{Y_c^{(i)}}^T \in \mathbf{R}^{n\times n}.$$

Each covariance matrix $\Sigma_c^{(i)}$ is positive definite since $t$ is typically much larger than $n$. Denoting $N_c$ as the number of trials for condition $c$, the average covariance matrix is

$$\overline{\Sigma}_c=\frac{1}{N_c}\sum_{i=1}^{N_c}Y_c^{(i)}{Y_c^{(i)}}^T, \qquad \mbox{for }c\in\{+,-\}.$$

### Principles of CSP

The goal of CSP is to find filters such that the variance of the spatially filtered signals under one condition is minimized, while that of the other condition is maximized. Specifically, denoting spatial filters as $X\in\mathbf{R}^{n\times n}$ (each column corresponding to a spatial filter), the covariance matrix of the filtered signals in class is given by

$$X^TY_c^{(i)}{Y_c^{(i)}}^T X=X^T\Sigma_c^{(i)}X.$$

Collectively, the covariance matrix of all the spatially filtered signals $Y_c^{(i)}$ in class $c$ is taken as 

$$X^T\overline{\Sigma}_cX.$$

CSP imposes the following couple of constraints on the spatial filters:

1. $X^T\overline{\Sigma}\_{-}X = \Lambda\_{-}$ and $X^T\overline{\Sigma}\_{+}X = \Lambda\_{+}$ such that $\Lambda\_{-}$ and $\Lambda\_{+}$ are diagonal matrices.

2. $X^T(\overline{\Sigma}\_{-} + \overline{\Sigma}\_{+})X=\Lambda\_{-}+\Lambda\_{+}=I_n$ where $I_n$ denotes the $n \times n$ identity matrix.

### Equivalence to a generalized eigenvalue problem and a generalized Rayleigh quotient optimization

These two constraints imply that the spatial filters correspond to the eigenvectors of the generalized eigenvalue problem

$$\overline{\Sigma}_-x = \lambda (\overline{\Sigma}_-+\overline{\Sigma}_+)x.$$

