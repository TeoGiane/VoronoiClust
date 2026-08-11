#' Generate Multi-View Data
#'
#' @description
#' Generates synthetic multi-view clustering data with an arbitrary number of views,
#' configurable noise levels, and controlled cluster agreement across views.
#' The generation strictly adheres to the procedure of retaining exact fractions
#' of cluster assignments while permuting the remainder, preserving marginal
#' cluster proportions across all views.
#'
#' @param n_obs Integer. The total number of observations (data points) to generate.
#' @param n_vars Integer. The number of variables (size of each data point) in each view [Default: 1].
#' @param n_clusters Integer. The number of latent clusters [Default: 2].
#' @param n_views Integer. The number of data views to generate [Default: 1].
#' @param dirichlet_conc Numeric. The concentration parameter for the Dirichlet distribution used to generate base cluster probabilities [Default: 1.0].
#' @param simplex_radius Numeric. The distance of each cluster center from the origin along orthogonal axes, defining the vertices of a regular simplex. Controls the baseline separation between clusters (distance between centres is \eqn{r\sqrt{2}} [Default: 1.0].
#' @param noise_sd Numeric vector or scalar. The standard deviation of the Gaussian noise added to the points. If a scalar is provided, the same noise level is applied to all views. If a vector, it must have length `n_views` [Default: 0.1].
#' @param agreement_rate Numeric vector or scalar. The exact fraction (between 0 and 1) of observations in views 2 through `n_views` that retain their cluster assignment from the first view. If a scalar is provided, the same agreement rate is applied to all views. If a vector, it must have length `n_views - 1`. Controls the correlation/agreement between views [Default: 0.0].
#' @param random_seed Integer. Optional seed for reproducibility [Default: NULL].
#' @param base_clusters Integer vector. Optional pre-specified cluster assignments for the first view. Must have length `n_obs`. If NULL, assignments are generated automatically [Default: NULL].
#'
#' @return A list containing the generated multi-view data and ground truth:
#' \describe{
#'   \item{points}{A list of length `n_views`, where each element is an `n_obs` x `n_vars` matrix of data points.}
#'   \item{clusters}{A list of length `n_views`, where each element is an integer vector of length `n_obs` representing cluster assignments.}
#'   \item{distances}{A list of length `n_views`, where each element is an `n_obs` x `n_obs` distance matrix.}
#'   \item{probs}{A numeric vector of length `n_clusters` containing the underlying cluster probabilities.}
#'   \item{centres}{An `n_clusters` x `n_vars` matrix of the base cluster centres.}
#' }
#'
#' @export
generate_multiview_data <- function(n_obs, n_vars = 1, n_clusters = 2, n_views = 1, dirichlet_conc = 1.0, simplex_radius = 1.0, noise_sd = 0.1, agreement_rate = 0, random_seed = NULL, base_clusters = NULL) {
  # Input validation
  stopifnot(n_obs >= 1, n_clusters >= 1, n_clusters <= n_obs,
            dirichlet_conc > 0, n_vars >= n_clusters,
            simplex_radius > 0, n_views >= 1)

  # Allow scalar or vector inputs for noise_sd and agreement_rate
  if (length(noise_sd) == 1) noise_sd <- rep(noise_sd, n_views)
  if (length(agreement_rate) == 1) agreement_rate <- rep(agreement_rate, n_views - 1)

  stopifnot(length(noise_sd) == n_views)
  stopifnot(length(agreement_rate) == n_views - 1)

  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }

  # 1. Generate base cluster probabilities and View 1 assignment
  probs <- MCMCpack::rdirichlet(1, rep(dirichlet_conc, n_clusters))
  if (is.null(base_clusters)) {
    base_clusters <- sort(sample(1:n_clusters, n_obs, replace = TRUE, prob = probs))
  }

  # 2. Generate cluster assignments for all views (Star Topology linked to View 1)
  clusts_list <- vector("list", n_views)
  clusts_list[[1]] <- base_clusters
  
  if (n_views >= 2) {
    for (v in 2:n_views) {
      # 2a. Determine the exact number of objects to keep identical based on floor()
      n_keep <- floor(n_obs * agreement_rate[v - 1])
  
      # 2b. Randomly select WHICH exact objects keep their assignment
      keep_idx <- sample(1:n_obs, size = n_keep, replace = FALSE)
  
      clusts_v <- rep(0, n_obs)
  
      # Assign the retained clusters
      if (n_keep > 0) {
        clusts_v[keep_idx] <- base_clusters[keep_idx]
      }
  
      # 2c. For the remaining objects, assign a random permutation of the REMAINING labels.
      change_idx <- setdiff(1:n_obs, keep_idx)
      if (length(change_idx) > 0) {
        clusts_v[change_idx] <- sample(base_clusters[change_idx], replace = FALSE)
      }
  
      clusts_list[[v]] <- clusts_v
    }
  }

  # 3. Define cluster centres (orthogonal vectors in 'n_vars' space)
  clust_centres <- matrix(0, nrow = n_clusters, ncol = n_vars)
  for (i in 1:n_clusters) {
    clust_centres[i, ] <- c(rep(0, i - 1), simplex_radius, rep(0, n_vars - i))
  }

  # 4. Generate points and distance matrices (Vectorized)
  points_list <- vector("list", n_views)
  dist_list <- vector("list", n_views)

  for (v in 1:n_views) {
    # Map each point to its assigned cluster center
    centres_assigned <- clust_centres[clusts_list[[v]], ]

    # Vectorized noise generation
    noise <- matrix(rnorm(n_obs * n_vars, mean = 0, sd = noise_sd[v]),
                    nrow = n_obs, ncol = n_vars)

    # Final points for view v
    pnts <- centres_assigned + noise

    points_list[[v]] <- pnts
    dist_list[[v]] <- as.matrix(dist(pnts))
  }

  return(list(
    points = points_list,
    clusters = clusts_list,
    distances = dist_list,
    probs = probs,
    centres = clust_centres
  ))
}
