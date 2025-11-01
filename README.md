#  Interaction-Based Clustering (IBC) — A Step-by-Step Article

### Abstract
**Interaction-Based Clustering (IBC)** is a practical and theoretical framework linking important interaction features (products of variables like $X_i \times X_j$) to natural clusters in data. When an interaction dominates the data-generating process, one variable of that interaction (or a combination of them) can define regime boundaries. Splitting data along those regimes and fitting local models often yields large improvements in predictive performance and interpretability. This article builds the idea from first principles, provides three mathematical proofs, demonstrates them with reproducible concepts, extends the concept to higher-degree interactions, and outlines how to apply IBC to high-dimensional real-world data.

---

## Table of Contents
* [1. Notation & Setup](#1-notation--setup)
* [2. The Central Idea (Informal)](#2-the-central-idea-informal)
* [3. Proof 1 — Piecewise Linear / Conditional-Slope Argument](#3-proof-1--piecewise-linear--conditional-slope-argument)
* [4. Proof 2 — Geometric Interpretation (Surface Twisting)](#4-proof-2--geometric-interpretation-surface-twisting)
* [5. Proof 3 — Variance Decomposition & Residual Conditioning](#5-proof-3--variance-decomposition--residual-conditioning)
* [6. Expected $R^2$ Improvement from Clustering (Intuition & Formula)](#6-expected-r2-improvement-from-clustering-intuition--formula)
* [7. From Theory to Practice — Algorithmic IBC Pipeline](#7-from-theory-to-practice--algorithmic-ibc-pipeline)
* [8. Simulations and Experiments (Conceptual Demonstration)](#8-simulations-and-experiments-conceptual-demonstration)
* [9. When D₁/D₂ can beat global interaction model C](#9-when-d₁d₂-can-beat-global-interaction-model-c)
* [10. High-Dimensional & Real-World Considerations](#10-high-dimensional--real-world-considerations)
* [11. Practical Recommendations & Diagnostics](#11-practical-recommendations--diagnostics)
* [12. Conclusion & Future Directions](#12-conclusion--future-directions)

---

## 1. Notation & Setup

| Term | Notation | Description |
| :--- | :--- | :--- |
| **Features** | $X_1, X_2, \dots, X_p$ | The set of predictor variables. |
| **Target** | $Y$ | The outcome variable. |
| **2nd Order Interaction** | $X_i \times X_j$ | Product of two predictors. |
| **$d$-th Order Interaction** | $\prod_{m=1}^d X_{k_m}$ | Product of $d$ predictors. |

We denote a **dominant interaction** as one with large absolute coefficient in the true generating process or in a selected model (e.g., large Lasso coefficient).

**Models compared:**
* **A** — Linear model on original features (no interactions).
* **B (IBC)** — Split dataset by a rule derived from an interaction variable (e.g., sign or quantile of $X_j$), fit separate linear regressions per cluster, combine predictions.
* **C** — Global linear model including the interaction feature(s) (e.g., include $X_i X_j$).
* **D₁/D₂** — Fit interaction models inside each cluster (i.e., per-cluster model that also uses interaction terms).

---

## 2. The Central Idea (Informal)
If the data-generating process includes a strong interaction term $\beta_{ij} (X_i X_j)$, then for fixed $X_j=c$ we have
$$\mathbb{E}[Y \mid X_i, X_j=c] \approx \beta_{ij} c \cdot X_i + \text{(small terms)}.$$
So **conditioning on $X_j$ transforms the multiplicative effect into a linear relationship in $X_i$ with slope proportional to $c$**. If $X_j$ takes different typical values in different regions (e.g., positive vs negative, low vs high), these regions are natural clusters where the $X_i$-to-$Y$ relationship is approximately linear but with different slopes. Clustering along $X_j$ therefore turns a global nonlinear interaction into locally linear relationships, enabling simple models to perform much better.

---

## 3. Proof 1 — Piecewise Linear / Conditional-Slope Argument
**Statement**
Given
$$Y = \beta_0 + \beta_i X_i + \beta_j X_j + \beta_{ij}(X_i X_j) + \varepsilon,$$
the conditional slope of $Y$ w.r.t. $X_i$ is
$$\frac{\partial \mathbb{E}[Y \mid X_j]}{\partial X_i} = \beta_i + \beta_{ij} X_j.$$
Hence, fixing $X_j$ (or splitting $X_j$ into regions) gives different linear relationships in $X_i$. If $|\beta_{ij}|$ is large relative to $|\beta_i|$, then the slope sign/direction changes with $X_j$.

**Consequence**
Split data by threshold(s) on $X_j$ (e.g., sign or median), fit separate linear models in each split $\to$ those piecewise linear models capture most of the interaction effect.

---

## 4. Proof 2 — Geometric Interpretation (Surface Twisting)
**Statement**
Consider $(X_i, X_j, Y)$ space. Without the interaction term the expected response surface is planar:
$$Y = \beta_0 + \beta_i X_i + \beta_j X_j.$$
Adding $\beta_{ij} X_i X_j$ produces a saddle-like or twisted surface:
$$Y = \beta_0 + \beta_i X_i + \beta_j X_j + \beta_{ij} X_i X_j.$$
The sign of $X_i X_j$ partitions the domain into different lobes of the surface, which correspond to distinct regimes. Each lobe can be approximated well by a local plane (a cluster-specific linear model). 

---

## 5. Proof 3 — Variance Decomposition & Residual Conditioning
**Statement**
Total variance:
$$\mathrm{Var}(Y) = \mathrm{Var}\big( \mathbb{E}[Y \mid C] \big) + \mathbb{E}\big[ \mathrm{Var}(Y \mid C) \big]$$
If we set $C$ as a clustering indicator (e.g., sign of $X_j$), and the dominant portion of $\mathrm{Var}(Y)$ arises from the interaction $\beta_{ij} X_i X_j$, then $\mathrm{Var}\big( \mathbb{E}[Y \mid C] \big)$ will be large, and within-cluster residual variance will be comparatively small. Therefore clustering explains a large share of variance and improves $R^2$.

---

## 6. Expected $R^2$ Improvement from Clustering (Intuition & Formula)
If for cluster $k$ we have (approximately) a linear model with variance explained $R^2_k$, and cluster weight $w_k$, the combined clustered $R^2$ is
$$R^2_{\text{clustered}} = \sum_k w_k R^2_k.$$
If the interaction explains variance proportional to $\beta_{ij}^2 \cdot \mathrm{Var}(X_i X_j)$, then clustering that isolates directions where $X_j$ (or $X_i$) takes distinct values converts multiplicative variance into linear explained variance, increasing per-cluster $R^2$. The expected improvement scales with $\beta_{ij}^2 \cdot \mathrm{Var}(X_i X_j)$ and with how well the cluster rule separates distinct conditional means.

---

## 7. From Theory to Practice — Algorithmic IBC Pipeline
A compact practical pipeline:

* **Feature engineering:** standardize numeric variables; create candidate interactions up to desired degree (use `interaction_only=True` to avoid powers if desired).
* **Screen / select interactions:** use $\text{LassoCV} / \text{ElasticNetCV}$ on standardized interaction features to find top interactions.
* **Select clustering variable(s):** choose one variable from the top interaction (e.g., for $X_i X_j$ choose $X_j$, or evaluate both). Optionally consider joint clustering if interaction is multi-variable.
* **Define split rule:** sign/median/quantile/tree split on selected variable(s).
* **Fit per-cluster models:** simple linear regression or models that include interaction features.
* **Evaluate:** compare A (no interaction global), C (global with interaction), B (clustered simple), D (clustered models including interactions) via cross-validated metrics.

---

## 8. Simulations and Experiments (Conceptual Demonstration)
To demonstrate the concepts, a synthetic dataset is generated where the target variable ($Y$) is strongly driven by an interaction term, $Y \propto 3.0 \cdot (X_i X_j)$.

### 8.1 Data Generation
A dataset is created where the dominant interaction is $X_i X_j$.
```python
import numpy as np, pandas as pd
np.random.seed(42)
n = 1000
Xi = np.random.normal(0,1,n)
Xj = np.random.normal(0,1,n)
beta0, betai, betaj, betaij = 0.5, 0.1, 0.1, 3.0
Y = beta0 + betai*Xi + betaj*Xj + betaij*(Xi*Xj) + np.random.normal(0,1,n)
df = pd.DataFrame({'Xi': Xi, 'Xj': Xj, 'Y': Y})
```

### 8.2 Compare models A, B (IBC), C
**Model A (Linear, No Interaction):** Fitting a global linear regression on $X_i$ and $X_j$ alone results in a low $R^2$ score (e.g., $\approx 0.1$).


**Model B (IBC Clustered):** Splitting the data based on the sign of $X_j$ (the clustering variable) and fitting separate linear regressions for each cluster significantly improves the fit (e.g., $R^2 \approx 0.7$). This is the core IBC gain.

**Model C (Global with Interaction):** Fitting a single global linear regression that includes the explicit interaction term $X_i X_j$ achieves the best fit for this simulated data (e.g., $R^2 \approx 0.85$), serving as a benchmark.
```python
### 8.2 Compare models A, B (IBC), C

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# A: global linear without interaction
X_A = df[['Xi','Xj']]
y = df['Y'].values
modelA = LinearRegression().fit(X_A, y)
r2A = r2_score(y, modelA.predict(X_A))

# B: IBC - split by Xj >= 0
mask = df['Xj'] >= 0
preds = np.empty(len(df))
model1 = LinearRegression().fit(X_A[mask], y[mask])
preds[mask] = model1.predict(X_A[mask])
model2 = LinearRegression().fit(X_A[~mask], y[~mask])
preds[~mask] = model2.predict(X_A[~mask])
r2B = r2_score(y, preds)

# C: global linear with interaction
X_C = np.column_stack([df['Xi'], df['Xj'], df['Xi']*df['Xj']])
modelC = LinearRegression().fit(X_C, y)
r2C = r2_score(y, modelC.predict(X_C))

print("R² A (no interaction):", r2A)
print("R² B (IBC clustered):", r2B)
print("R² C (with interaction):", r2C)
```
### 8.3 D₁/D₂: Fit interaction models inside clusters
**Models D₁/D₂ (Clustered Interaction Models):** If we split the data by the cluster rule and then fit models *that also include the interaction term* inside each cluster, the fit can be slightly better or equal to Model C in this homogeneous simulation, but this structure becomes crucial when the true coefficients or noise levels vary by regime.
```python
# D models (per-cluster interaction models)
mask = df['Xj'] >= 0
# D1
X1 = np.column_stack([df.loc[mask,'Xi'], df.loc[mask,'Xj'], df.loc[mask,'Xi']*df.loc[mask,'Xj']])
y1 = df.loc[mask, 'Y'].values
mD1 = LinearRegression().fit(X1, y1)
# D2
X2 = np.column_stack([df.loc[~mask,'Xi'], df.loc[~mask,'Xj'], df.loc[~mask,'Xi']*df.loc[~mask,'Xj']])
y2 = df.loc[~mask, 'Y'].values
mD2 = LinearRegression().fit(X2, y2)

predsD = np.empty(len(df))
predsD[mask] = mD1.predict(X1)
predsD[~mask] = mD2.predict(X2)
r2D = r2_score(y, predsD)
print("R² D (clustered interaction models):", r2D)

```
---

## 9. When D₁/D₂ can beat global interaction model C
D₁/D₂ can outperform C when:

* **Regime-dependent coefficients:** $\beta_{ij}$ (or other coefficients) differ significantly between clusters.
* **Regime-specific intercepts or omitted variables:** each regime has unique baseline or unobserved features.
* **Heteroskedasticity:** different noise levels across regimes.
* **Nonlinearities only active in certain regimes:** global polynomial fails to capture localized nonlinearity.
* **Finite-sample / regularization effects:** splitting reduces variance of parameter estimates relative to a single global model that over-regularizes.

**Diagnosis:** fit C and D on held-out data (or CV). If D has higher out-of-sample $R^2$ or lower error, the data are heterogeneous and regime models are warranted.

---

## 10. High-Dimensional & Real-World Considerations
When $p$ is large (genomics, finance, ads):

* **Feature explosion:** number of interactions grows combinatorially. Use **screening** (univariate filters, domain priors) to limit candidates.
* **Regularization:** use $\text{LassoCV}, \text{ElasticNetCV}$, or hierarchical/group penalties (Group Lasso) to select interaction groups.
* **Clustering rule complexity:** interactions of degree $>2$ often require multivariate cluster rules (small decision trees, $k$-means, mixture models) rather than a simple single-variable sign split.
* **Stability & interpretability:** trees or sparse methods help find stable splits and interpretable clusters.
* **Cross-validation:** essential to validate whether clustered models generalize better.

**Domain-specific notes:**
* **Finance:** cluster on volatility quantiles or regime indicators discovered by interactions of macro variables.
* **Genomics:** screen SNPs/genes by marginal tests, then investigate pairwise/triple interactions for epistasis; cluster on gene expression patterns.
* **Marketing / Social:** interactions between channel $\times$ demographic $\times$ time can reveal segments for targeted models.

---

## 11. Practical Recommendations & Diagnostics
* **Start small:** try degree-2 interactions first. Use `interaction_only=True`.
* **Standardize** raw features before building polynomial features (important for Lasso).
* **Screen** candidates (univariate correlations, domain knowledge) to reduce search.
* When top interaction $X_i X_j$ found, **visualize $X_i$ vs $Y$ colored by $X_j$** (or vice versa). If branches appear, IBC likely helps.
* **Compare models A/B/C/D on held-out sets** (not just in-sample $R^2$).
* If D wins but cluster sizes are small, consider hierarchical or mixture-of-experts models to borrow strength.
* **Automate selection** of the clustering variable by scanning candidate variables in top interactions and comparing cross-validated gains in $R^2$.

---

## 12. Conclusion & Future Directions
IBC connects model interpretability, feature selection, and clustering: important interaction features reveal regime structure; splitting on those regimes often converts a global nonlinear relationship into locally linear pieces that are easier to model and interpret.

**Key takeaways:**
* If interaction truly generates the signal and is homogeneous, **model C (global interaction) is optimal**.
* If interaction behavior or noise differs by regime, **cluster-specific interaction models (D) can beat global models**.
* In high-dimensional settings, combine screening + sparse selection + careful cross-validation.

**Future work:** rigorous statistical tests for when to prefer D over C (penalized likelihood criteria, hierarchical Bayes), automatic multi-variable split discovery, and scalable IBC for ultra-high-dimensional data.

---

## 13. Appendix — compact code summary
Below is a compact snippet that performs the full automated IBC discovery (selection → cluster → compare models). It is a simplified blueprint — tune and extend for production.

```python
# Compact IBC pipeline (simplified)
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score

def ibc_pipeline(df, target, degree=2, top_k=5):
    X = df.drop(columns=[target])
    y = df[target].values
    # 1. Create interaction-only features and names
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    Xpoly = poly.fit_transform(X)
    names = poly.get_feature_names_out(X.columns)
    Xpoly_df = pd.DataFrame(Xpoly, columns=names)
    # 2. Standardize and LassoCV
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xpoly_df)
    lasso = LassoCV(cv=5, random_state=0).fit(Xs, y)
    coefs = pd.Series(np.abs(lasso.coef_), index=names).sort_values(ascending=False)
    top = coefs.head(top_k).index.tolist()
    print("Top interactions:", top)
    # 3. Choose variable from the top interaction (last token)
    candidate = top[0].split()[-1]  # crude choice
    print("Clustering by:", candidate)
    mask = df[candidate] >= df[candidate].median()
    # 4. Fit clustered linear models (no interactions)
    model1 = LinearRegression().fit(X[mask], y[mask])
    model2 = LinearRegression().fit(X[~mask], y[~mask])
    preds = np.empty(len(y))
    preds[mask] = model1.predict(X[mask])
    preds[~mask] = model2.predict(X[~mask])
    r2_clustered = r2_score(y, preds)
    # 5. Fit global interaction model (use Xpoly_df)
    modelC = LinearRegression().fit(Xpoly_df, y)
    r2_global = r2_score(y, modelC.predict(Xpoly_df))
    return {'r2_clustered': r2_clustered, 'r2_global_interaction': r2_global, 'top_interactions': top}

# Usage:
# result = ibc_pipeline(df, 'Y', degree=2)
# print(result)

```
