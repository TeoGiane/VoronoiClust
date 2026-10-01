# VoronoiClust

`VoronoiClust` is an `R` package for Bayesian distance clustering of (multi-view) data. It provides reversible-jump MCMC for tessellation models and split-merge MCMC for Pitman–Yor process mixture models. Both model families can also be fit jointly across multiple views.

The samplers accept distance matrices rather than raw feature matrices, so they can be used with distances chosen for the data and domain at hand.

## Installation

Install the package from a local source checkout with:

```r
install.packages("remotes")
remotes::install_local("path/to/VoronoiClust")
```

The package contains compiled C++ code. Make sure `R` build tools and the system dependencies required by `RcppGSL` are available on your platform.

## Quick start

This example creates a small two-group dataset, computes Euclidean distances, estimates likelihood parameters, and runs the tessellation sampler.

```r
library("VoronoiClust)"

set.seed(42)
n_per_group <- 30
points <- rbind(
	cbind(rnorm(n_per_group, mean = -2), rnorm(n_per_group)),
	cbind(rnorm(n_per_group, mean =  2), rnorm(n_per_group))
)
distance_matrix <- as.matrix(dist(points))

likelihood <- compute_EB_params(
	distance_matrix,
	n_clusters = 2,
	linear = FALSE
)

prior <- list(
	type = "truncated_geometric",
	min = 1,
	max = nrow(distance_matrix),
	prob = 0.25
)

algorithm <- list(
	iterations = 2000,
	burnin = 500,
	thinning = 5
)

fit <- mcmc_tessellation(distance_matrix, likelihood, prior, algorithm)

str(fit)
```

`fit$cluster_allocs` contains the retained cluster assignments, `fit$n_clust` the number of clusters in each retained sample, and `fit$lpdf` the corresponding log-posterior values. Use `?mcmc_tessellation` for the complete return-value and parameter documentation.

## Available models

- `mcmc_tessellation()` runs reversible-jump MCMC for tessellation models. It uses a truncated-geometric prior on the number of clusters.
- `mcmc_PY()` runs split-merge MCMC for a Pitman–Yor process mixture. The prior can be fixed (`type = "PY-fixed"`) or hierarchical (`type = "PY-hierarchical"`).
- `mcmc_tessellation_multiview()` and `mcmc_PY_multiview()` fit the corresponding models to a list of distance matrices, each representing a view.

All samplers take an algorithm-parameter list. Common settings include `iterations`, `burnin`, `thinning`, and `random_seed`; model-specific settings are described in `?algorithm_params`.

For the single-view samplers, `compute_EB_params()` creates likelihood parameters from a distance matrix using an initial k-medoids partition. Set `linear = TRUE` for the faster medoid-based likelihood, or leave it `FALSE` to estimate from all within- and between-cluster pairwise distances. You can also provide your own likelihood parameter list; see `?likelihood_params`.

## Multiple views

Pass one square distance matrix per view to a multiview sampler. Likelihood parameters are organized per view, alongside coupling hyperparameters; prior parameters are organized per view:

```r
likelihood_params <- list(
	views_params = list(likelihood_view_1, likelihood_view_2),
	coupling_params = list(strength_alpha = 1, strength_beta = 1)
)

prior_params <- list(
	views_params = list(prior_view_1, prior_view_2)
)

fit <- mcmc_tessellation_multiview(
	distance_matrices,
	likelihood_params,
	prior_params,
	algorithm
)
```

The multiview result contains a `views` list with each view's sampler output and a `joint_lpdf` vector for the joint log-posterior. See `?multiview_params` for the parameter structure.

## Synthetic data

`generate_multiview_data()` creates synthetic points, cluster assignments, and pairwise distance matrices for one or more views. Its result includes `points`, `clusters`, `distances`, cluster probabilities, and cluster centres. Set `random_seed` for reproducibility; `agreement_rate` controls the fraction of assignments retained between the first view and each subsequent view.

## Further documentation

- `?compute_EB_params` for empirical-Bayes likelihood initialization.
- `?likelihood_params`, `?prior_params`, and `?algorithm_params` for sampler configuration.
- `?mcmc_PY` and `?mcmc_tessellation` for single-view samplers.
- `?mcmc_PY_multiview`, `?mcmc_tessellation_multiview`, and `?multiview_params` for multiview fitting.
- `?generate_multiview_data` for synthetic data generation.
