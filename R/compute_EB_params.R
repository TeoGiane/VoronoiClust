#' Compute Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters for the likelihood model from data using k-medoids clustering.
#'
#' @param distance_matrix A square numeric matrix of pairwise distances between data points.
#' @param n_clusters The number of clusters for the initial k-medoids clustering. This is ignored if `initial_allocs` is provided.
#' @param initial_allocs Optional initial cluster assignments. If \code{NULL} (the default), k-medoids clustering is performed using `n_clusters`.
#' @param linear Logical. If \code{TRUE}, uses a faster estimation method based on distances to/between medoids. If \code{FALSE} (the default), uses all pairwise distances, which is more accurate but slower.
#' @param repulsion Logical. If \code{TRUE} (the default), the returned parameters will indicate that the model should account for between-cluster repulsion.
#'
#' @return A list of likelihood parameters in the format required by \code{mcmc_tessellation} and \code{mcmc_PY}. See \code{\link{likelihood_params}}.
#' @export
#' @importFrom fastkmedoids fastpam
compute_EB_params <- function(distance_matrix, n_clusters, initial_allocs = NULL, linear = FALSE, repulsion = TRUE){
  if(linear){
    return(compute_EB_params_linear(D = distance_matrix, K = n_clusters, k2 = initial_allocs, repulsion = repulsion))
  } else {
    return(compute_EB_params_quadratic(D = distance_matrix, K = n_clusters, k2 = initial_allocs, repulsion = repulsion))
  }
}


#' Linear Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters using a "linear" complexity approach (distances to medoids).
#'
#' @inheritParams compute_EB_params
#' @return A list of likelihood parameters.
#' @keywords internal
compute_EB_params_linear <- function(D, K, k2 = NULL, repulsion = TRUE) {
  n <- nrow(D)
  if(is.null(k2)) {
    medoidsfit <- cluster::clara(D, K)
    k2 <- medoidsfit$clustering
    medoids <- medoidsfit$i.med
    # medoidsfit <- fastkmedoids::fastpam(D, n, K)
    # k2         <- medoidsfit@assignment
    # medoids    <- medoidsfit@medoids + 1        #they are using C++
  } else {
    r_D                 <- rowSums(D)
    names(r_D)          <- 1:n
    k2_unique           <- unique(k2)
    medoids             <- numeric(length(k2_unique))
    for(i in 1:length(unique(k2))){
      medoids[i] <- as.numeric(names(which.min(r_D[k2==k2_unique[i]])))
    }
  }

  #within distances------------------------------------------------------------------------
  within_distance <- c()
  for(k in 1:length(medoids)){
    label_k = k2[medoids[k]]
    within_distance <- c(within_distance, D[medoids[k], k2==label_k])
  }
  within_distance   <- within_distance[within_distance>0]

  ##find hyperprior for within-cluster parameters
  shape_within <- mean(within_distance)^2/var(within_distance)
  prior_shape_within <- shape_within*length(within_distance)
  prior_rate_within  <- sum(within_distance)

  #between distances------------------------------------------------------------------------
  D_medoids <- D[medoids,medoids]
  between_distance <- D_medoids[upper.tri(D_medoids)]

  ##find hyperprior for between-cluster parameters (Gamma)
  shape_between <- mean(between_distance)^2/var(between_distance) # Corresponds to delta2
  rate_between  <- mean(between_distance)/var(between_distance)   # Corresponds to theta

  return(list(
    type = "linear",
    shape_within = shape_within,
    prior_shape_within = prior_shape_within,
    prior_rate_within = prior_rate_within,
    shape_between = shape_between, # delta2
    rate_between = rate_between,   # theta
    repulsion = repulsion
  ))
}


#' Quadratic Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters using a "quadratic" complexity approach (all pairwise distances).
#'
#' @inheritParams compute_EB_params
#' @return A list of likelihood parameters.
#' @keywords internal
compute_EB_params_quadratic <- function(D, K, k2 = NULL, repulsion = TRUE) {
  n <- nrow(D)
  if(is.null(k2)){
    medoidsfit <- cluster::clara(D, K)
    k2 <- medoidsfit$clustering
    # medoidsfit <- fastkmedoids::fastpam(D, n, K)
    # k2         <- medoidsfit@assignment
  }

  sel1 = outer(k2,k2, function(x,y) x==y) & upper.tri(diag(length(k2)))
  within_distance  <- na.omit(D[sel1])
  within_distance   <- within_distance[within_distance>0]

  sel2 = outer(k2,k2, function(x,y) x!=y) & upper.tri(diag(length(k2)))
  between_distance <- na.omit(D[sel2])
  between_distance <- between_distance[between_distance>0]

  ##find hyperprior for within-cluster parameters
  shape_within <- mean(within_distance)^2/var(within_distance)
  prior_shape_within <- shape_within*length(within_distance)
  prior_rate_within  <- sum(within_distance)

  ##find hyperprior for between-cluster parameters
  shape_between <- mean(between_distance)^2/var(between_distance)
  prior_shape_between <- shape_between*length(between_distance)
  prior_rate_between  <- sum(between_distance)

  return(list(
    type = "quadratic",
    shape_within = shape_within,
    prior_shape_within = prior_shape_within,
    prior_rate_within = prior_rate_within,
    shape_between = shape_between,
    prior_shape_between = prior_shape_between,
    prior_rate_between = prior_rate_between,
    repulsion = repulsion
  ))
}
