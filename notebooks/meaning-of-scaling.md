## `vegan::scores()` and scaling (1, 2, 3)

### 1. General context

All ordination methods in **vegan** — PCA, CA, RDA, CCA, dbRDA, etc. — produce two sets of coordinates:

* **Sites** (rows of the data matrix ( Y ))
* **Species** (columns of ( Y ))

They originate from a **bilinear decomposition**:
[
Y_c = U \Sigma V^T
]
where ( Y_c ) is the centered (and possibly standardized) data matrix.

* ( U ) = site loadings (orthonormal)
* ( V ) = species loadings (orthonormal)
* ( \Sigma ) = diagonal matrix of singular values

The eigenvalues ( \lambda_i = \Sigma_{ii}^2 / (n - 1) ) quantify the variance explained by axis ( i ).

All scaling types are simply **rescalings of ( U \Sigma ) and ( V )** so that distances or correlations have ecological meaning.

---

### 2. The three common scaling options in *vegan*

| Scaling | Name (Legendre & Legendre)        | Goal                                  | Site coordinates   | Species coordinates |
| ------- | --------------------------------- | ------------------------------------- | ------------------ | ------------------- |
| 1       | Species scaling / distance biplot | Preserve inter-site distances         | ( U \Sigma )       | ( V )               |
| 2       | Site scaling / correlation biplot | Preserve correlations between species | ( U )              | ( V \Sigma )        |
| 3       | Symmetric scaling                 | Equal weight to sites and species     | ( U \Sigma^{1/2} ) | ( V \Sigma^{1/2} )  |

All three obey the general constraint:
[
Y_c \approx (\text{site scores}) \times (\text{species scores})^T
]
but differ in how variance is split between rows and columns.

---

### 3. Detailed formulas

Given SVD ( Y_c = U \Sigma V^T ) and eigenvalues ( \lambda_i = \Sigma_{ii}^2 / (n - 1) ):

#### Scaling 1 — “distance biplot”

Distances among sites approximate Euclidean distances among rows of ( Y_c ).

[
\begin{aligned}
\text{Site scores} &= U \Sigma \
\text{Species scores} &= V
\end{aligned}
]

* Preserves **inter-site distances**.
* The origin is the centroid of sites.
* Arrows (species) show correlation with ordination axes.

#### Scaling 2 — “correlation biplot”

Angles among species vectors approximate correlations among species.

[
\begin{aligned}
\text{Site scores} &= U \
\text{Species scores} &= V \Sigma
\end{aligned}
]

* Preserves **species correlations**.
* Often used for interpreting which species drive gradients.

#### Scaling 3 — “symmetric scaling”

A compromise: both sites and species are scaled equally.

[
\begin{aligned}
\text{Site scores} &= U \Sigma^{1/2} \
\text{Species scores} &= V \Sigma^{1/2}
\end{aligned}
]

* Equalizes variance between rows and columns.
* Used when both sets should be displayed comparably in the same biplot.

---

### 4. Generalization to constrained ordinations (RDA, CCA)

For **RDA (redundancy analysis)**:

* Compute fitted values ( \hat{Y} = H Y_c ) using the hat matrix ( H = X_c (X_c^T X_c)^{-1} X_c^T ).
* Perform SVD: ( \hat{Y} = U_c \Sigma_c V_c^T ).
* Apply the same scaling rules to ( U_c, \Sigma_c, V_c ).

For **CCA (canonical correspondence)**:

* Replace SVD with weighted SVD using chi-square distances:
  [
  D_r^{-1/2} (Y / \mathbf{1}^T Y - \mathbf{r}\mathbf{c}^T) D_c^{-1/2} = U \Sigma V^T
  ]
  where ( D_r, D_c ) are diagonal matrices of row and column sums.
* Apply scaling on the weighted coordinates similarly.

Thus, all scaling choices derive from how we post-multiply ( U, V ) by powers of ( \Sigma ):
[
\text{Sites} = U \Sigma^a,\quad \text{Species} = V \Sigma^{1-a}
]
where ( a = 1, 0, \tfrac{1}{2} ) correspond to scaling 1, 2, 3 respectively.

---

### 5. Algorithmic summary (language-neutral)

**Inputs:**

* Data matrix ( Y ) (n × p)
* Optional predictors ( X ) (for RDA/CCA)
* Boolean `center` and `standardize`

**Steps:**

1. Center (and optionally standardize) columns of ( Y ).
2. Compute fitted values ( \hat{Y} ) if constrained, else ( Y_c ).
3. Run SVD: ( \hat{Y} = U \Sigma V^T ).
4. Compute eigenvalues ( \lambda = \Sigma^2 / (n - 1) ).
5. Derive site and species coordinates via:

   ```text
   scaling 1: sites = U * Σ, species = V
   scaling 2: sites = U, species = V * Σ
   scaling 3: sites = U * sqrt(Σ), species = V * sqrt(Σ)
   ```
6. Optionally, scale axes so that `sum(sites^2) = n - 1` if matching R’s normalization.

---

### 6. Implementation notes

* When matching **`vegan`**, remember:

  * R uses column-centered data by default.
  * Eigenvalues are normalized by ( n - 1 ).
  * For constrained ordination, the residual space and fitted space are orthogonal and handled separately.

* Species and site scaling apply identically to **PCA**, **RDA**, and **CCA**, differing only in how ( Y ) is pre-processed and what metric defines distances (Euclidean for PCA/RDA, chi-square for CCA).

---

### 7. Minimal Python sketch

```python
def vegan_scaling(U, s, Vt, scaling=1):
    if scaling == 1:
        sites = U * s
        species = Vt.T
    elif scaling == 2:
        sites = U
        species = Vt.T * s
    elif scaling == 3:
        sites = U * np.sqrt(s)
        species = Vt.T * np.sqrt(s)
    else:
        raise ValueError("Scaling must be 1, 2, or 3.")
    return sites, species
```