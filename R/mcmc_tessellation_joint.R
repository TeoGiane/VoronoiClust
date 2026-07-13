#' Joint MCMC Sampler for Multi-View Voronoi Tesselation
#'
#' Performs Markov Chain Monte Carlo (MCMC) sampling over Voronoi tesselation 
#' configurations for an arbitrary number of coupled views. The function iterates 
#' through each view sequentially, proposing updates to cluster centers and 
#' accepting or rejecting them via a Metropolis-Hastings step that incorporates 
#' both the marginal likelihood and a generalized pairwise coupling term.
#'
#' @param D_list A list of length V containing the distance matrices for each view, 
#'   where $V$ is the number of views and each matrix is of dimension N times N.
#' @param prior_param Numeric; prior parameter controlling the distribution of the 
#'   number of clusters. Default is 3.
#' @param params A nested list of model parameters. Must contain `hyperparams_lik` 
#'   (a list of length $V$ with view-specific likelihood hyperparameters, including 
#'   `medoids.init` for initial cluster centers). It must also contain parameters 
#'   required by the coupling function (e.g., `alpha.a`, `alpha.b`).
#' @param N.sim Integer; the number of MCMC iterations to perform. Default is 10000.
#' @param tempering Numeric; tempering parameter for the proposal distribution. 
#'   Default is 0.
#' @param verbose Logical; if TRUE, prints iteration progress and details. 
#'   Default is FALSE.
#'
#' @return A list of length `N.sim` containing the MCMC samples. Each element is a 
#'   list with the following components:
#'   \describe{
#'     \item{\code{z}}{A list of length $V$ containing the cluster allocations 
#'       (tesselations) for each view.}
#'     \item{\code{K}}{An integer vector of length $V$ detailing the current number 
#'       of unique clusters in each view.}
#'     \item{\code{centers}}{A list of length $V$ containing the indices of the 
#'       cluster centers for each view.}
#'     \item{\code{log_lik}}{Numeric; the sum of the marginal log-likelihoods 
#'       across all views for the current iteration.}
#'     \item{\code{time}}{An object of class \code{proc_time} tracking the 
#'       cumulative computational time elapsed.}
#'   }
#'
#' @export
mcmc_tesselation_joint <- function(D_list, prior_param = 3, params, N.sim = 10000, tempering = 0, verbose = FALSE) {
  # print("mcmc_tesselation_joint")
  # Deduce number of views and number of data
  V <- length(D_list)
  N <- nrow(D_list[[1]])
  
  # Initialize State Lists
  centers <- vector("list", V)
  gamma <- vector("list", V)
  z <- vector("list", V)
  log.mlik <- vector("numeric", V)
  
  # Clean params and initialize for each view
  for (v in 1:V) {
    params$hyperparams_lik[[v]][["within_distance"]]  <- NULL # NON SON SICURO SERVA (+ check per iperparametri)
    params$hyperparams_lik[[v]][["between_distance"]] <- NULL # NON SON SICURO SERVA (+ check per iperparametri)
    centers[[v]] <- params$hyperparams_lik[[v]]$medoids.init
    gamma[[v]] <- rep(0, N)
    gamma[[v]][centers[[v]]] <- 1
    z[[v]] <- tesselation(D_list[[v]], centers[[v]])
    log.mlik[v] <- get_log_prob_D_tesselation(D_list[[v]], z[[v]], centers[[v]], params$hyperparams_lik[[v]])
  }
  
  # Create samples buffer
  samples <- vector("list", N.sim)
  
  # Set progress bar
  cat(sprintf("  VoronoiClust: Multiview Tessellation MCMC (%d iterations)\n", N.sim))
  pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 3, width = 50, char = "*")
  start = proc.time()
  
  # MCMC Loop
  for (i in 1:N.sim) {
    
    # Loop over each view to perform Gibbs/Metropolis updates
    for (v in 1:V) {
      
      # Compute and extract proposed state
      proposal <- generate_gamma_proposal(i, D_list[[v]], gamma[[v]], centers[[v]], z[[v]], prior_param, tempering, params$hyperparams_lik[[v]])
      gamma.new    <- proposal$gamma.new  # Note: Ensure your proposal func naming is consistent!
      centers.new  <- proposal$centers.new # In your orig code, gamma2 proposal extraction differed from gamma1
      z.new        <- proposal$z.new
      log.mlik.new <- proposal$log.mlik.new
      prob_new_old <- proposal$prob_new_old
      prob_old_new <- proposal$prob_old_new
      
      # Calculate generalized coupling term
      coupling_new <- compute_conditional_coupling_logloss(v, z.new, z, params)
      # coupling_new <- calc_generalized_coupling(v, z.new, z, D_list, params)
      coupling_old <- compute_conditional_coupling_logloss(v, z[[v]], z, params)
      # coupling_old <- calc_generalized_coupling(v, z[[v]], z, D_list, params)
      
      ### Accept/Reject
      numerator <- log.mlik.new + log_prior_k(length(centers.new), N, prior_param) + log(prob_old_new) + coupling_new
      denominator <- log.mlik[v] + log_prior_k(length(centers[[v]]), N, prior_param) + log(prob_new_old) + coupling_old
      
      if (runif(1) < exp(numerator - denominator)) {
        z[[v]]       <- z.new
        centers[[v]] <- centers.new
        gamma[[v]]   <- gamma.new
        log.mlik[v]  <- log.mlik.new
      }
    }
    
    setTxtProgressBar(pb, i)
    
    # Save current sample: store everything as a list of lists
    samples[[i]] <- list(
      z = z,
      K = sapply(z, function(x) length(unique(x))),
      centers = centers,
      log_lik = sum(log.mlik),
      time = difftime(proc.time()["elapsed"], start["elapsed"], units = "secs")
    )
  }
  close(pb)
  
  # Print total elapsed time in seconds
  elapsed_time <- samples[[length(samples)]]$time
  cat(sprintf("  Elapsed Time: %g s\n", elapsed_time))
  
  return(samples)
}


#' Generate Gamma Proposal for Birth-Death-Move
#'
#' Generates a proposal for Voronoi center configuration using birth, death, or move moves.
#'
#' @param i Iteration number (determines move type)
#' @param D Distance matrix
#' @param gamma Binary indicator of center status
#' @param centers Current center indices
#' @param z Current cluster assignments
#' @param prior_param Prior parameters
#' @param tempering Tempering parameter
#' @param params Model parameters
#'
#' @return List with proposed configuration and transition probabilities
#' @keywords internal
generate_gamma_proposal <- function(i,  D, gamma, centers, z, prior_param, tempering,  params) {
  # print("generate_gamma_proposal()")
  K <- length(centers)                        #number of centers
  N <- length(gamma)
  #if no cluster or cluster saturation then cannot move... then birth or death
  if(i %% 2 ==1 || K==0 || K==N){             #BIRTH OR DEATH STEP -------------------------
    birth = (runif(1)>0.5 || K < 2) & (K!=N) # prob > 0.5 or birth if no clusters and no cluster saturation -> if not then death!
    
    #print(birth)
    if(birth){
      new_probs <- get_probs("birth", D, gamma, centers, z, prior_param, tempering,  params)
    }else{
      new_probs <- get_probs("death", D, gamma, centers, z, prior_param, tempering,  params)
    }
    
    
    flip.index                 <- sample(1:N, 1, prob = new_probs)         #informed proposal...
    gamma.new                  <- gamma
    gamma.new[flip.index]      <- 1 - gamma.new[flip.index]
    centers.new                <- which(gamma.new == 1)
    z.new                      <- tesselation(D, centers.new)
    log.mlik.new               <- get_log_prob_D_tesselation(D, z.new, centers.new,  params)
    
    #get transition probs
    if(birth){
      prob_old_new <- get_probs("death", D, gamma.new, centers.new, z.new, prior_param, tempering,  params)[flip.index]
    }else{
      prob_old_new <- get_probs("birth", D, gamma.new, centers.new, z.new, prior_param, tempering,  params)[flip.index]
    }
    
    prob_new_old   <- new_probs[flip.index]
    
  }
  
  else{ # MOVE STEP ------------------------------------
    
    move_index <- sample(centers, 1)
    new_probs  <- get_probs("move", D,  gamma, centers, z, prior_param, tempering,  params, move_index = move_index) #pick one at random and choose
    
    flip.index                 <- sample(1:N, 1, prob = new_probs)         #informed proposal...
    gamma.new                  <- gamma
    gamma.new[move_index]      <- 0
    gamma.new[flip.index]      <- 1
    centers.new                <- which(gamma.new == 1)
    z.new                      <- tesselation(D, centers.new)
    log.mlik.new               <- get_log_prob_D_tesselation(D, z.new, centers.new,  params)
    
    #get transition probabilities
    prob_old_new <- get_probs("move", D,  gamma.new, centers.new, z.new, prior_param, tempering,  params, move_index = flip.index)[move_index]
    prob_new_old   <- new_probs[flip.index]
  }
  
  return(list(gamma.new = gamma.new, centers.new = centers.new,
              z.new = z.new, log.mlik.new = log.mlik.new,
              prob_new_old = prob_new_old, prob_old_new = prob_old_new))
}


#' Helper Functions for Rand Index
#'
#' Check if two elements are in the same cluster.
#'
#' @param pair Numeric vector of length 2 with cluster labels
#' 
#' @return Logical value
#' 
#' @keywords internal
f_same <- function(pair) {
  pair[1] == pair[2]
}


#' Helper Functions for Rand Index
#'
#' Check if two elements are in different clusters.
#'
#' @param pair Numeric vector of length 2 with cluster labels
#'
#' @return Logical value
#' 
#' @keywords internal
f_diff <- function(pair) {
  pair[1] != pair[2]
}


#' Compute Rand Index Between Two Clusterings
#'
#' Computes the Rand index agreement measure between two clustering solutions.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#'
#' @return Numeric value of Rand index
#' @keywords internal
rand_index <- function(z1, z2) {

  # Total number of pairs
  n <- length(z1)
  total_pairs <- n*(n-1)/2

  # Creating a matrix to compare each pair for both clusterings
  agree_same <- combn(z1, 2, FUN = f_same) & combn(z2, 2, FUN = f_same)
  agree_diff <- combn(z1, 2, FUN = f_diff) & combn(z2, 2, FUN = f_diff)

  # Summing agreements (both in the same cluster and in different clusters)
  TP_and_TN <- sum(agree_same) + sum(agree_diff)

  # Since each pair is counted twice, we need to divide by 2
  rand_index <- TP_and_TN / total_pairs

  return(rand_index)
}


#' Compute conditional coupling log-loss
#' 
#' Computes the conditional coupling log-loss between the clustering in the current view and the clustering in all other views.
#' 
#' @param curr_view Integer; the index of the view currently being updated
#' @param target_z Numeric vector; the proposed (or current) tessellation for view \code{curr_view}
#' @param z_list List of numeric vectors; the current tessellations for ALL views
#' @param params List of model hyperparametes (here uses alpha.a and alpha.b)
#' 
#' @return A single numeric value representing the total condition coupling log-loss
#' @keywords internal
compute_conditional_coupling_logloss <- function(curr_view, target_z, z_list, params) {
  # Detect number of views
  V <- length(z_list)
  # Initialize total log-loss
  log_coupling_sum <- 0
  # Compute constants
  log_const <- lgamma(params$alpha.a) - lbeta(params$alpha.a, params$alpha.b)
  for (u in 1:V) {
    # Skip comparing the view to itself
    if (u == curr_view) {
      next
    }
    # Compute distance between the two tessellations
    dist_val <- 1.0 / rand_index(target_z, z_list[[u]]) - 1.0
    # Calculate the log of the Tricomi function
    log_tricomi <- log(GPBayes::HypergU(params$alpha.a, 1 - params$alpha.b, dist_val))
    # Add to the running sum
    log_coupling_sum <- log_coupling_sum + (log_const + log_tricomi)
  }
  # return
  return(log_coupling_sum)
}



## ORIGINAL CODE (ONLY TWO VIEWS)

###################################################################################
################## joint sampler ##################################################
###################################################################################




#' Compute Coupling Log Likelihood
#'
#' Computes the log likelihood for the coupling between two clusterings using Hypergeometric function.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param D2 Distance matrix (unused)
#' @param a Shape parameter for beta distribution
#' @param b Shape parameter for beta distribution
#'
#' @return Log likelihood
#' @keywords internal
# C_lik_log <- function(z1, z2, D2 = NULL, a, b){
#   #d       = 2/(salso::ARI(z2,z2.proj)+1)-1
#   d <- 1/rand_index(z1,z2)-1
#   return( lgamma(a) - lbeta(a,b) + log(GPBayes::HypergU(a, 1-b,d)))
# }


#' Sample Alpha Conditional on Clusterings
#'
#' Generates a sample from the conditional distribution of alpha given two clusterings using inverse transform sampling.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param D2 Distance matrix
#' @param a Shape parameter
#' @param b Shape parameter
#'
#' @return Numeric sample from conditional alpha distribution
#' @keywords internal
# alpha_conditional_sample <- function(z1, z2, D2, a, b){
# 
#   #z2.proj = tesselation(D2, medoids1)
#   #d       = 2/(salso::ARI(z2,z2.proj)+1)-1
#   d        = 1/rand_index(z1,z2)-1
# 
#   xmin <- 1/10000
#   xmax <- 1-1/10000
#   N    <- 1000
#   alpha   <- seq( xmin, xmax,,N)
#   phi     <- alpha/(1-alpha)
#   log_pdf <-  -phi*d +(b-1)*log(1-alpha) +(a-1)*log(alpha)
#   pdf     <- unlogprob_to_prob(log_pdf)
#   cdf     <- cumsum(pdf)
#   quant   <- splinefun(x = cdf, y = alpha, ties = "mean")
#   sample  <- quant(runif(1))
#   if(sample<0) sample = xmin
#   if(sample>1) sample = xmax
#   return(sample)
# }


#' Extract Alpha Samples from MCMC Output
#'
#' Extracts conditional samples of alpha parameter from MCMC chain output.
#'
#' @param fit List of MCMC samples from mcmc_tesselation_joint
#' @param D2 Second distance matrix
#' @param a Shape parameter
#' @param b Shape parameter
#' @param N Number of samples to extract (if NULL, uses full chain length)
#'
#' @return Numeric vector of alpha samples
#' @keywords internal
# get_samples_alpha <- function(fit, D2, a, b, N = NULL){
# 
#   if(is.null(N)){ N = length(fit)}
# 
#   check.seq <- round(seq(1,length(fit),,N)) #indeces to check
# 
#   alpha <- numeric(N)
#   for(i in 1:N){
#     alpha[i] <- alpha_conditional_sample(fit[[i]]$z1,  fit[[i]]$z2, D2, a, b)
#   }
#   return(alpha)
# }


#' Joint MCMC Sampler for Two-Layer Voronoi Tesselation
#'
#' Performs MCMC sampling over Voronoi tesselation configurations for two coupled distance matrices.
#'
#' @param D1 First distance matrix
#' @param D2 Second distance matrix
#' @param prior_param Prior parameter for number of clusters (default 3)
#' @param params List of model parameters including hyperparam_lik_1 and hyperparam_lik_2
#' @param N.sim Number of MCMC iterations (default 10000)
#' @param tempering Tempering parameter (default 0)
#' @param verbose Logical; print iteration details (default FALSE)
#'
#' @return List of MCMC samples with z1, z2, centers, and log likelihood
#' @keywords internal
# mcmc_tesselation_joint  <- function(D1, D2, prior_param = 3, params,
#                                     N.sim = 10000, tempering = 0,
#                                     verbose = FALSE){
# 
#   params$hyperparam_lik_1[["within_distance"]]  <- NULL
#   params$hyperparam_lik_1[["between_distance"]] <- NULL
#   params$hyperparam_lik_2[["within_distance"]]  <- NULL
#   params$hyperparam_lik_2[["between_distance"]] <- NULL
# 
#   N <- nrow(D1)
# 
#   #variables for layer 1
#   centers1           <- params$hyperparam_lik_1$medoids.init
#   gamma1             <- rep(0, N)
#   gamma1[centers1]   <- 1
#   z1                 <- tesselation(D1, centers1)
#   log.mlik1          <- get_log_prob_D_tesselation(D1, z1, centers1,  params$hyperparam_lik_1)
# 
#   centers2           <- params$hyperparam_lik_2$medoids.init
#   gamma2             <- rep(0, N)
#   gamma2[centers2]   <- 1
#   z2                 <- tesselation(D2, centers2)
#   log.mlik2          <- get_log_prob_D_tesselation(D2, z2, centers2,  params$hyperparam_lik_2)
# 
#   samples <- list()
#   pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 0)
#   start = proc.time()
#   for(i in 1:N.sim){
# 
#     ##Proposal for gamma1-------------------------------------------------------
#     proposal       <- generate_gamma_proposal(i, D1, gamma1, centers1, z1, prior_param, tempering,  params$hyperparam_lik_1)
#     gamma1.new     <- proposal$gamma.new
#     centers1.new   <- proposal$centers.new
#     z1.new         <- proposal$z.new
#     log.mlik1.new  <- proposal$log.mlik.new
#     prob_new_old   <- proposal$prob_new_old
#     prob_old_new   <- proposal$prob_old_new
# 
#     ###accept/reject
#     numerator   <- log.mlik1.new + log_prior_k(length(centers1.new), N, prior_param) + log(prob_old_new) + C_lik_log(z1.new, z2, D2, params$alpha.a, params$alpha.b)
#     denominator <- log.mlik1 + log_prior_k(length(centers1),N,prior_param)  + log(prob_new_old) + C_lik_log(z1, z2, D2, params$alpha.a, params$alpha.b)
#     acceptance_prob <-  exp( numerator - denominator )
# 
# 
# 
#     if(runif(1) < acceptance_prob){
#       z1       = z1.new
#       centers1 = centers1.new
#       gamma1   = gamma1.new
#       log.mlik1 = log.mlik1.new
#     }
# 
# 
#     ##Proposal for gamma2-------------------------------------------------------
#     proposal     <- generate_gamma_proposal(i, D2, gamma2, centers2, z2, prior_param, tempering,  params$hyperparam_lik_2)
#     gamma2.new   <- proposal$gamma
#     prob_new_old <- proposal$prob_new_old
#     prob_old_new <- proposal$prob_old_new
#     centers2.new                <- which(gamma2.new == 1)
#     z2.new                      <- tesselation(D2, centers2.new)
#     log.mlik2.new               <- get_log_prob_D_tesselation(D2, z2.new, centers2.new,  params$hyperparam_lik_2)
# 
#     ###accept/reject
#     numerator   <- log.mlik2.new + log_prior_k(length(centers2.new), N, prior_param) + log(prob_old_new)  + C_lik_log(z1, z2.new, D2, params$alpha.a, params$alpha.b)
#     denominator <- log.mlik2 + log_prior_k(length(centers2),N,prior_param)  + log(prob_new_old)  + C_lik_log(z1, z2, D2, params$alpha.a, params$alpha.b)
#     acceptance_prob <-  exp( numerator - denominator )
# 
#     #
#     #+
# 
#     if(runif(1) < acceptance_prob){
#       z2       = z2.new
#       centers2 = centers2.new
#       gamma2   = gamma2.new
#       log.mlik2 = log.mlik2.new
#     }
# 
#     setTxtProgressBar(pb,i)
#     if(verbose){
#       print(paste0("iteration ", i))
#       print(centers1)
#       print(centers2)
#     }
# 
#     samples[[length(samples)+1]] <- list(z1=z1, z2 = z2,
#                                          K1 = length(unique(z1)), K2 = length(unique(z2)),
#                                          centers1 = centers1, centers2 = centers2,
#                                          log_lik = log.mlik1 + log.mlik2,
#                                          time = proc.time() - start)
# 
#   }
#   close(pb)
# 
# 
#   return(samples)
# }