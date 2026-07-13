
#' Update M Parameter in Marginal MCMC Sampler
#'
#' Performs a Metropolis-Hastings step to update the M parameter using truncated normal proposals.
#'
#' @param z Numeric vector of cluster assignments
#' @param M Current value of M parameter
#' @param theta Current value of theta parameter
#' @param params List of model parameters including metropolis_M_sd and lambda.M
#' @param rcpp Logical; whether to use Rcpp implementations
#'
#' @return Updated value of M parameter
#' @keywords internal
update_M_marginal <- function(z, M, theta, params, rcpp){
  # list2env(params,.GlobalEnv)
  
  candidate_M   <- truncnorm::rtruncnorm(1, a=0, mean = M, sd = params$metropolis_M_sd)
  
  
  candidate_log_prob <-  get_log_prob_partition(z, candidate_M, theta,rcpp) + dexp(candidate_M, params$lambda.M, log = TRUE)
  old_log_prob       <-  get_log_prob_partition(z, M, theta, rcpp) + dexp(M, params$lambda.M, log = TRUE)
  
  log_transition_prob_old_new <- log(truncnorm::dtruncnorm(M, mean = candidate_M, sd = params$metropolis_M_sd))
  log_transition_prob_new_old <- log(truncnorm::dtruncnorm(candidate_M, mean = M, sd = params$metropolis_M_sd))
  
  accept_ratio  <- exp(log_transition_prob_old_new - log_transition_prob_new_old +  candidate_log_prob - old_log_prob)
  if(runif(1)<=accept_ratio){
    return(candidate_M)
  }
  else{
    return(M)
  }
} 


#' Update Theta Parameter in Marginal MCMC Sampler
#'
#' Performs a Metropolis-Hastings step to update the theta parameter using truncated normal proposals.
#'
#' @param z Numeric vector of cluster assignments
#' @param M Current value of M parameter
#' @param theta Current value of theta parameter
#' @param params List of model parameters
#' @param rcpp Logical; whether to use Rcpp implementations
#'
#' @return Updated value of theta parameter
#' @keywords internal
update_theta_marginal <- function(z,  M, theta, params,rcpp){
  # list2env(params,.GlobalEnv)
  candidate_theta   <- truncnorm::rtruncnorm(1, a=0, b=1, mean = theta, sd = params$metropolis_theta_sd)
  
  
  candidate_log_prob <-  get_log_prob_partition(z, M, candidate_theta,rcpp)+ dbeta(candidate_theta, params$alpha.theta, params$beta.theta, log = TRUE)
  old_log_prob       <-  get_log_prob_partition(z, M, theta,rcpp) + dbeta(theta, params$alpha.theta, params$beta.theta, log = TRUE)
  
  log_transition_prob_old_new <- log(truncnorm::dtruncnorm(theta, mean = candidate_theta, sd = params$metropolis_theta_sd))
  log_transition_prob_new_old <- log(truncnorm::dtruncnorm(candidate_theta, mean = theta, sd = params$metropolis_theta_sd))
  
  accept_ratio  <- exp(log_transition_prob_old_new - log_transition_prob_new_old + candidate_log_prob - old_log_prob)

  if(runif(1)<=accept_ratio){
    return(candidate_theta)
  }
  else{
    return(theta)
  }
} 



#' Compute Log Probabilities for Cluster Allocations
#'
#' Calculates log probabilities for assigning observation i to each possible cluster.
#'
#' @param z Numeric vector of cluster assignments
#' @param i Integer index of observation to compute probabilities for
#' @param M Parameter M
#' @param theta Parameter theta
#' @param D Distance matrix or data matrix
#' @param params List of model parameters
#' @param rcpp Logical; whether to use Rcpp implementations (default TRUE)
#'
#' @return Numeric vector of log probabilities
#' @keywords internal
get_log_prob_allocations_marginal <- function(z, i, M, theta, D, params, rcpp = TRUE){
  if(rcpp){
    return(get_log_prob_allocations_marginal_rcpp(z, i, M, theta, D, params))
  }  
  else{
    # list2env(params,.GlobalEnv)
    K_mi       <- length(unique(z[-i]))
    log_probs  <- numeric(K_mi+1) 
    n          <- length(z)
    
    
    #probability of being assigned to existing clusters
    for(k in 1:(length(log_probs))){
      z_temp    <- z
      z_temp[i] <- k
      log_probs[k] <- get_log_prob_partition(z_temp,M,theta,rcpp) + get_log_prob_D(D, z_temp, params,rcpp)
    }
    
    #print(get_log_prob_partition(z_temp,M,theta))
    #print( get_log_prob_D(D,z_temp,params))
    return(log_probs)
  }
}

#' MCMC Sampler for Distance Clustering
#'
#' Main MCMC sampler for Bayesian Distance Clustering with Pitman-Yor process priors.
#' Performs inference on cluster allocations, the mass parameter M, and the discount parameter theta.
#'
#' @param D Distance or data matrix (n x n matrix)
#' @param params List of MCMC parameters including:
#'   \itemize{
#'     \item z.init: Initial cluster allocations
#'     \item M.init: Initial value for M parameter
#'     \item theta.init: Initial value for theta parameter
#'     \item metropolis_M_sd: Proposal standard deviation for M
#'     \item metropolis_theta_sd: Proposal standard deviation for theta
#'     \item lambda.M: Rate parameter for exponential prior on M
#'     \item alpha.theta, beta.theta: Shape parameters for beta prior on theta
#'     \item delta1, mu, beta: Hyperparameters for within-cluster distances
#'     \item delta2, zeta, gamma: Hyperparameters for between-cluster distances
#'   }
#' @param N.sim Integer number of MCMC iterations to run
#' @param verbose Logical; if TRUE, print iteration details (default FALSE)
#' @param rcpp Logical; if TRUE, use Rcpp implementations for speed (default TRUE)
#' @param M.fixed Optional fixed value for M parameter; if NULL, M is sampled
#' @param theta.fixed Optional fixed value for theta parameter; if NULL, theta is sampled
#'
#' @return List of samples, each containing:
#'   \itemize{
#'     \item M: Sampled M value
#'     \item theta: Sampled theta value
#'     \item z: Cluster allocations
#'     \item K: Number of clusters
#'     \item log_lik: Log-likelihood for WAIC
#'     \item time: Elapsed time
#'   }
#'
#' @examples
#' \dontrun{
#' # Set up parameters for a distance matrix D
#' params <- list(
#'   z.init = rep(1, nrow(D)),
#'   M.init = 10,
#'   theta.init = 0.5,
#'   metropolis_M_sd = 1,
#'   metropolis_theta_sd = 0.1,
#'   lambda.M = 1,
#'   alpha.theta = 1,
#'   beta.theta = 1,
#'   delta1 = 1, mu = 1, beta = 1,
#'   delta2 = 1, zeta = 1, gamma = 1
#' )
#' # Run sampler for 1000 iterations
#' samples <- mcmc_sampler(D, params, N.sim = 1000)
#' }
#'
#' @export
mcmc_sampler <- function(D, params, N.sim, verbose=FALSE, rcpp = TRUE, M.fixed = NULL, theta.fixed = NULL) {
  
  # if(is.null(params$repulsion)){
  #   params$repulsion = FALSE
  # }
  
  # list2env(params,.GlobalEnv)
  n <- nrow(D)
  
  z <- params$z.init
  
  if(is.null(M.fixed)) {
    M = params$M.init
  } else {
    M = M.fixed
  }
  
  if(is.null(theta.fixed)) {
    theta = params$theta.init
  } else {
    theta = theta.fixed
  }
  
  samples <- list()
  cat(sprintf("  VoronoiClust: Pitman-Yor MCMC (%d iterations)\n", N.sim))
  pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 3, width = 50, char = "*")
  start = proc.time()
  for(i in 1:N.sim) {
      
    for(j in 1:n){
      log_probs <- get_log_prob_allocations_marginal(z,j,M,theta,D,params,rcpp)
      z <- update_z(z, j, log_probs, rcpp)
    }
    
    # update M
    if(is.null(M.fixed)) {
      M <- update_M_marginal(z, M, theta, params, rcpp) 
    }
    
    #update theta
    if(is.null(theta.fixed)) {
      theta <- update_theta_marginal(z, M, theta, params, rcpp) 
    }
    
    #get log-likelihood for WAIC
    log_lik <- get_log_prob_D(D, z, params, rcpp)
    
    samples[[length(samples)+1]] <- list(M = M, theta=theta, z = z, K = length(unique(z)), log_lik = log_lik,
                                         time = difftime(proc.time()["elapsed"], start["elapsed"], units = "secs"))
    
    setTxtProgressBar(pb,i)
    if(verbose){
      print(paste0("iteration ", i))
      print(z)
      print(M)
      print(theta)
    }
  }  
  close(pb)
  
  # Print total elapsed time in seconds
  elapsed_time <- samples[[length(samples)]]$time
  cat(sprintf("  Elapsed Time: %g s\n", elapsed_time))
  
  return(samples)
}





