#' @name algorithm_params
#' @title Algorithm Parameters
#' @description A list specifying the parameters for the MCMC algorithm.
#'
#' @section Tessellation Model (`mcmc_tessellation`):
#' \itemize{
#'   \item `iterations`: Total number of MCMC iterations.
#'   \item `burnin`: Number of burn-in iterations.
#'   \item `thinning`: Thinning interval.
#'   \item `init_n_clust`: Initial number of clusters.
#'   \item `tempering`: Tempering parameter for proposal distribution (0 for uniform, >0 for informed).
#'   \item `random_seed`: Seed for the random number generator.
#'   \item `debug`: A boolean to enable debug messages.
#' }
#'
#' @section Mixture Model (`mcmc_PY`):
#' \itemize{
#'   \item `iterations`: Total number of MCMC iterations.
#'   \item `burnin`: Number of burn-in iterations.
#'   \item `thinning`: Thinning interval.
#'   \item `init_n_clust`: Initial number of clusters.
#'   \item `n_sweeps`: Number of restricted Gibbs sweeps in Split-Merge moves.
#'   \item `target_acc_rate`: Target acceptance rate for adaptive Metropolis-Hastings updates of hyperparameters.
#'   \item `adapt_decay`: Decay rate for the Robbins-Monro adaptation of proposal standard deviation.
#'   \item `random_seed`: Seed for the random number generator.
#'   \item `debug`: A boolean to enable debug messages.
#' }
NULL