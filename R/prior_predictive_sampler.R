#' Sample from Pitman-Yor Process with Gaussian base distribution
#'
#' Generates samples from a Pitman-Yor process using stick-breaking construction.
#'
#' @param N Number of samples
#' @param alpha Mass parameter  
#' @param theta Discount parameter
#' @param n_approx Number of approximation terms
#'
#' @return Numeric vector of cluster assignments
#' @export
sample_from_pitman_yor_process <- function(N, alpha, theta, n_approx=10000) {
  # Sample weights from a beta distribution with parameters alpha and d
  beta_samples <- rbeta(n_approx, 1 - theta, alpha + theta * seq(0, n_approx - 1))
  
  # Stick-breaking process
  weights <- exp(log(beta_samples) + c(log(1), cumsum(log(1 - beta_samples[-n_approx]))))
  
  # Sample from the Pitman-Yor Process
  sampled_data <- sample(1:n_approx, size = N, replace = TRUE, prob = weights)
  
  #make sure it is 1,1,1,1,2,2,2,....
  # Create a dictionary to map original values to new values
  unique_values <- unique(sampled_data)
  new_values <- 1:length(unique_values)
  rearranged_vector <- match(sampled_data, unique_values)
  
  #print(sum(weights))
  if(sum(weights)<0.99) warning("Sampling from PY in Stick-breaking manner not accurate")
  return(rearranged_vector)
}


#' Marginal Prior Predictive Sampler
#'
#' Samples from the prior predictive distribution (without observed data).
#'
#' @param n Number of observations
#' @param N.sim Number of samples to generate
#' @param params Model parameters
#' @param M.fixed Optional fixed M value
#' @param theta.fixed Optional fixed theta value
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return List of samples with M, theta, z, D, and distance summaries
#' @export
marginal_prior_predictive_sampler <- function(n, N.sim, params, M.fixed = NULL, theta.fixed = NULL, rcpp = TRUE){
  #I can generate samples from lambda, theta, alpha, rho.... 
  #how to generate samples from D?
  #simulate lambda_k
  list2env(params,.GlobalEnv)
  samples <- list()
  
  for(l in 1:N.sim){
    
    if(!is.null(M.fixed)){
      M     = M.fixed
    }else{
      M     = rexp(1,lambda.M)
    }
    if(!is.null(theta.fixed)){
      theta = theta.fixed
    }else{
      theta = rbeta(1,alpha.theta, beta.theta)
    }
    
    z <- sort(sample_from_pitman_yor_process(n, M, theta))
    
    # Layer 0
    if(rcpp){
      dat <- sampleDistances_rcpp(z,params)
      D1 = dat$D
      within_distances_1 = dat$within_distances
      between_distances_1 = dat$between_distances
    }else{
      cluster_labels <- unique(z)
      n_clust = length(cluster_labels)
      lambda  <- rgamma(n_clust, params$mu, params$beta)
      theta   <- matrix(rgamma(n_clust^2, params$zeta, params$gamma), n_clust, n_clust)
      
      cluster_labels <- unique(z)
      D <- matrix(NA, n, n)
      within_distances_1  <- c()
      between_distances_1 <- c()
      for(i in 1:(n-1)){
        for(j in (i+1):n){
          if(z1[i]==z1[j]){
            D[i,j] = rgamma(1,params$delta1,lambda[which(z[i]==cluster_labels)])
            D[j,i] = D[i,j]
            within_distances_1 <- c(within_distances_1, D[i,j])
          }else{
            D[i,j] = rgamma(1,params$delta2,theta[which(z[i]==cluster_labels),which(z[j]==cluster_labels)])
            D[j,i] = D[i,j]
            between_distances_1 <- c(between_distances_1, D[i,j])
          }
        }
      }
      D1 = D
    }
    
    sample <- list(M = M, theta = theta,
                   z = z,
                   D1 = D1, 
                   within_distances_1 = within_distances_1,
                   between_distances_1 = between_distances_1)
    
    samples[[length(samples)+1]] <- sample 
    
    
    if(l %% 10 == 0) print(paste0("iteration ", l))
  }
  
  return(samples)
  
}


#' Prior Predictive Sampler
#'
#' Samples from the prior predictive distribution for two-layer clustering.
#'
#' @param n Number of observations
#' @param N.sim Number of samples to generate
#' @param params Model parameters with hyperparam_lik_1 and hyperparam_lik_2
#' @param alpha.fixed Optional fixed alpha value
#' @param M.fixed Optional fixed M value
#' @param theta.fixed Optional fixed theta value
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return List of samples with M, theta, alpha, gamma, z1, z2, and distance matrices
#' @export
prior_predictive_sampler <- function(n, N.sim, params, alpha.fixed = NULL, M.fixed = NULL, theta.fixed = NULL,  rcpp){
  #I can generate samples from lambda, theta, alpha, rho.... 
  #how to generate samples from D?
  #simulate lambda_k
  list2env(params,.GlobalEnv)
  samples <- list()
  
  gamma <- rep(0,n)
  z1 = rep(1,n)
  z2 = rep(1,n)
  
  for(l in 1:N.sim){
    
    if(!is.null(M.fixed)){
      M     = M.fixed
    }else{
      M     = rexp(1,lambda.M)
    }
    if(!is.null(theta.fixed)){
      theta = theta.fixed
    }else{
      theta = rbeta(1,alpha.theta, beta.theta)
    }
    if(!is.null(alpha.fixed)){
      alpha = alpha.fixed
    }else{
      alpha = rbeta(1,a_alpha, b_alpha)
    }
    
    #sample z1
    #for(i in 2:n){
    #  n_clust <- length(unique(z))
    #  
    #  prob    <- (plyr::count(z)$freq - theta)/(length(z) + M)
    #  prob    <- c(prob,(M + n_clust*theta)/(length(z) + M))
    #  z <- c(z,sample(1:(n_clust+1), size = 1, prob = prob))
    #}
    #z1=z
    #z2 = z1
    #update_indeces<- (1:n)[gamma==0]
    #z1 = rep(1,n)
    # for(j in 1:n){
    #   log_probs <- get_log_prob_allocations_prior(z1,j,M,theta,params)
    #   z1        <- update_z(z1,j,log_probs)
    # }
    # #print(z1)
    # 
    # #sample gamma
    # gamma <- rbinom(n, 1, alpha)
    # 
    # #sample z2 (fix this...)
    # update_indeces <- (1:n)[gamma==0]
    # #z2=z1
    # for(j in update_indeces){
    #   log_probs <- get_log_prob_allocations_prior(z2,j,M,theta,params)
    #   z2        <- update_z(z2,j,log_probs)
    # }
    #print(z2)
    #get_log_prob_partition(z1,M,theta)
    #get_log_prob_partition(z2,M,theta)
    
    #update z1
    for(j in 1:n){
      log_probs <- get_log_prob_allocations_comp_check(z1,z2,gamma,j,M,theta,matrix(NA,0,0),params,rcpp)
      #print(log_probs)
      z1        <- update_z(z1,j,log_probs)
    }
    
    #update gamma
    gamma <- update_gammas(matrix(NA,0,0), matrix(NA,0,0), z1, z2, M, theta, gamma, alpha, params,rcpp)
    
    
    #updating z2 
    #z2=z1 #CHECK IF THIS IS CORRECT
    update_indeces <- (1:n)[gamma==0]
    for(j in update_indeces){
      log_probs <- get_log_prob_allocations(z2,j,M,theta,matrix(NA,0,0),params,rcpp)
      z2        <- update_z(z2,j,log_probs)
    }
    
    # Layer 0
    if(rcpp){
      dat <- sampleDistances_rcpp(z1,params$hyperparam_lik_1)
      D1 = dat$D
      within_distances_1 = dat$within_distances
      between_distances_1 = dat$between_distances
    }else{
      cluster_labels <- unique(z1)
      n_clust = length(cluster_labels)
      lambda   <- rgamma(n_clust, params$hyperparam_lik_1$mu, params$hyperparam_lik_1$beta)
      theta  <- matrix(rgamma(n_clust^2, params$hyperparam_lik_1$zeta, params$hyperparam_lik_1$gamma), n_clust, n_clust)
      
      cluster_labels <- unique(z1)
      D <- matrix(NA, n, n)
      within_distances_1  <- c()
      between_distances_1 <- c()
      for(i in 1:(n-1)){
        for(j in (i+1):n){
          if(z1[i]==z1[j]){
            D[i,j] = rgamma(1,params$hyperparam_lik_1$delta1,lambda[which(z1[i]==cluster_labels)])
            D[j,i] = D[i,j]
            within_distances_1 <- c(within_distances_1, D[i,j])
          }else{
            D[i,j] = rgamma(1,params$hyperparam_lik_1$delta2,theta[which(z1[i]==cluster_labels),which(z1[j]==cluster_labels)])
            D[j,i] = D[i,j]
            between_distances_1 <- c(between_distances_1, D[i,j])
          }
        }
      }
      D1 = D
    }
    
    
    
    # Layer 1
    # Layer 0
    if(rcpp){
      dat <- sampleDistances(z2,params$hyperparam_lik_2)
      D2 = dat$D
      within_distances_2 = dat$within_distances
      between_distances_2 = dat$between_distances
    }else{
      cluster_labels <- unique(z2)
      n_clust = length(cluster_labels)
      lambda   <- rgamma(n_clust, params$hyperparam_lik_2$mu, params$hyperparam_lik_2$beta)
      theta  <- matrix(rgamma(n_clust^2, params$hyperparam_lik_2$zeta, params$hyperparam_lik_2$gamma), n_clust, n_clust)
      
      D <- matrix(NA, n, n)
      within_distances_2  <- c()
      between_distances_2 <- c()
      for(i in 1:(n-1)){
        for(j in (i+1):n){
          if(z2[i]==z2[j]){
            D[i,j] = rgamma(1,params$hyperparam_lik_2$delta1,lambda[which(z2[i]==cluster_labels)])
            D[j,i] = D[i,j]
            within_distances_2 <- c(within_distances_2, D[i,j])
          }else{
            #print(theta)
            #print(n_clust)
            #print(which(z2[i]==cluster_labels),which(z2[j]==cluster_labels))
            D[i,j] = rgamma(1,params$hyperparam_lik_2$delta2,theta[which(z2[i]==cluster_labels),which(z2[j]==cluster_labels)])
            #print(D[i,j])
            D[j,i] = D[i,j]
            between_distances_2 <- c(between_distances_2, D[i,j])
          }
        }
      }
      D2 = D
    }
    
    
    sample <- list(M = M, theta = theta, alpha = alpha,
                   z1 = z1, gamma = gamma, z2 = z2,
                   D1 = D1, D2 = D2, 
                   within_distances_1 = within_distances_1,
                   within_distances_2 = within_distances_2,
                   between_distances_1 = between_distances_1,
                   between_distances_2 = between_distances_2)
    
    samples[[length(samples)+1]] <- sample 
    
    
    if(l %% 10 == 0) print(paste0("iteration ", l))
  }
  
  return(samples)
  
}

