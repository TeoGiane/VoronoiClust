#' Get Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters from data using k-medoids clustering.
#'
#' @param D Distance matrix
#' @param k Number of clusters for initial k-medoids
#' @param k2 Optional cluster assignments; if NULL, k-medoids clustering is performed
#' @param linear Logical; if TRUE uses linear model, if FALSE uses quadratic (default FALSE)
#'
#' @return List of hyperparameters for the model
#' @keywords internal
get_EB_params <- function(D,k,k2=NULL, linear = FALSE){
  if(linear){
    return(get_EB_params_linear(D,k,k2))
  }else{
    return(get_EB_params_quadratic(D,k,k2))
  }
}

#' Linear Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters using linear model.
#'
#' @param D Distance matrix
#' @param K Number of clusters
#' @param k2 Optional cluster assignments; if NULL, k-medoids clustering is performed
#'
#' @return List of hyperparameters
#' @keywords internal
get_EB_params_linear <- function(D, K, k2=NULL){
  n <- nrow(D)
  if(is.null(k2)){
    medoidsfit <- fastkmedoids::fastpam(D, n, K)#, ...) #FCPS::kmeansClustering(D,K)$Cls    
    k2         <- medoidsfit@assignment
    medoids    <- medoidsfit@medoids + 1        #they are using C++ 
  }else{
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
    #print(sum(k2==label_k))
    within_distance <- c(within_distance, D[medoids[k], k2==label_k])
  }
  within_distance   <- within_distance[within_distance>0]

  
  ##find hyperprior delta1, mu, and beta
  delta1 <- mean(within_distance)^2/var(within_distance) #shape parameter of gamma fit of within distances... method of moments
  mu     <- delta1*length(within_distance)
  beta   <- sum(within_distance)

  #between distances------------------------------------------------------------------------
  D_medoids <- D[medoids,medoids]
  between_distance <- D_medoids[upper.tri(D_medoids)]
  
  ##find hyperprior delta2, zeta, and gamma 
  delta2 <- mean(between_distance)^2/var(between_distance)   #method of moments
  theta  <- mean(between_distance)/var(between_distance)
  #hist(between_distance)
  #hist(rgamma(1000,delta2,rgamma(1000,zeta,gamma)))
  
  return(list(delta1=delta1, mu=mu, beta=beta, delta2 = delta2, theta = theta,
              z.init = k2, medoids.init = medoids, within_distance = within_distance, between_distance = between_distance,
              linear=TRUE))
}

#' Quadratic Empirical Bayes Parameters
#'
#' Computes empirical Bayes hyperparameters using quadratic model.
#'
#' @param D Distance matrix
#' @param K Number of clusters
#' @param k2 Optional cluster assignments; if NULL, k-medoids clustering is performed
#'
#' @return List of hyperparameters
#' @keywords internal
get_EB_params_quadratic <- function(D, K, k2=NULL){
  n <- nrow(D)
  if(is.null(k2)){
    medoidsfit <- fastkmedoids::fastpam(D, n, K)    
    k2         <- medoidsfit@assignment
    medoids    <- medoidsfit@medoids + 1        #they are using C++ 
  }else{
    r_D                 <- rowSums(D)
    names(r_D)          <- 1:n
    k2_unique           <- unique(k2)
    medoids             <- numeric(length(k2_unique))
    for(i in 1:length(unique(k2))){
      medoids[i] <- as.numeric(names(which.min(r_D[k2==k2_unique[i]])))
    }
  }
  
  sel1 = outer(k2,k2, function(x,y) x==y) & upper.tri(diag(length(k2)))
  within_distance  <- na.omit(D[sel1])
  within_distance   <- within_distance[within_distance>0]
  
  sel2 = outer(k2,k2, function(x,y) x!=y) & upper.tri(diag(length(k2)))
  between_distance <- na.omit(D[sel2])
  between_distance <- between_distance[between_distance>0]
  
  ##find hyperprior delta1, mu, and beta
  delta1 <- mean(within_distance)^2/var(within_distance) #shape parameter of gamma fit of within distances... method of moments
  mu     <- delta1*length(within_distance)
  beta   <- sum(within_distance)
  
  #hist(within_distance)
  #hist(rgamma(1000,delta1,rgamma(1000,mu,beta)))
  
  ##find hyperprior delta2, zeta, and gamma 
  delta2 <- mean(between_distance)^2/var(between_distance)   #method of moments
  zeta   <- delta2*length(between_distance)
  gamma  <- sum(between_distance)
  
  #hist(between_distance)
  #hist(rgamma(1000,delta2,rgamma(1000,zeta,gamma)))
  
  return(list(delta1=delta1, mu=mu, beta=beta, delta2 = delta2, zeta = zeta,
              gamma = gamma, z.init = k2, medoids.init = medoids,
              linear=FALSE))
}

#' Within-Cluster Sum of Squares for K-medoids
#'
#' Plots WSS as function of number of clusters using fast k-medoids.
#'
#' @param D Distance matrix
#' @param Kmax Maximum number of clusters to evaluate
#'
#' @keywords internal
WSS_kmedoids <- function(D, Kmax){
  plot(2:Kmax,
       sapply(2:Kmax, function(x) fastkmedoids::fastpam(D, nrow(D), x)@cost),
       xlab = "K", 
       ylab=  "Cost")
}

#' Within-Cluster Sum of Squares for K-means
#'
#' Plots WSS as function of number of clusters using k-means.
#'
#' @param D Distance matrix
#' @param Kmax Maximum number of clusters to evaluate
#'
#' @keywords internal
WSS_kmeans <- function(D,Kmax){
  plot(2:Kmax,sapply(2:Kmax,function(x) sum(FCPS::kmeansClustering(D,x)$SSE)), xlab = "K", ylab="WSS")
}


#' Generate Two-Layer Mixture Data
#'
#' Generates synthetic data from a two-layer Gaussian mixture model.
#'
#' @param N Number of observations
#' @param K Number of clusters
#' @param M Mass parameter for Dirichlet prior
#' @param dim Dimension of data
#' @param radius Radius for cluster centers
#' @param sigma1 Standard deviation for layer 1
#' @param sigma2 Standard deviation for layer 2
#' @param alpha Dependence parameter between layers
#' @param seed Random seed
#' @param compute.oracle Logical; if TRUE compute oracle co-clustering matrices
#' @param rcpp Logical; use Rcpp implementations
#' @param clusts1 Optional cluster assignments for layer 1
#'
#' @return List containing generated data and cluster assignments
#' @keywords internal
generateMixture <- function(N, K, M = K, dim = K, radius = 1, sigma1 = 0.1, sigma2 = 0.1, alpha = 0, seed = NULL,
                           compute.oracle = FALSE, rcpp = TRUE, clusts1 = NULL) {
  # input validation 
  stopifnot(N >= 1, K >= 1, K <= N, M > 0, dim >= K, radius > 0, sigma1 > 0, sigma2 > 0)
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  #cluster assignment - first layer
  probs <- GPBayes::rdirichlet(1, rep(M, K))
  if(is.null(clusts1)){
    clusts1 <- sort(sample(1:K, N, replace = TRUE, prob = probs))
  }
  
  #cluster assignment - second layer
  clusts2           <- sample(clusts1)
  gamma             <- rbinom(N,1,alpha)
  clusts2[gamma==1] <- clusts1[gamma==1]
  
  # generate cluster centres for layer 1
  clust_centres <- matrix(0, nrow = K, ncol = dim)
  for (i in 1:K) {
    clust_centres[i,] <- c(rep(0, i - 1), radius, rep(0, dim - i))
  }
  
  # generate points layer 1
  pnts1 <- matrix(0, nrow = N, ncol = dim)
  Cov1 <- diag(rep(sigma1^2, dim))
  for (i in 1:N) {
    pnts1[i,] <- MASS::mvrnorm(1, clust_centres[clusts1[i],], Cov1)
  }
  
  # generate points layer 2
  pnts2 <- matrix(0, nrow = N, ncol = dim)
  Cov2 <- diag(rep(sigma2^2, dim))
  for (i in 1:N) {
    pnts2[i,] <- MASS::mvrnorm(1, clust_centres[clusts2[i],], Cov2)
  }
  
  # calculate oracle
  if(compute.oracle == TRUE){
    numiters <- 500
    
    ### LAYER 1
    oracle_posterior <- matrix(0, nrow = K, ncol = N)
    oracle_coclustering <- matrix(0, nrow = N, ncol = N)
    for (c in 1:numiters) {
      tempprobs <- rdirichlet(1, rep(M, K))
      
      if(rcpp){
        update_oracle_list  <- updateOracle_rcpp(oracle_posterior,oracle_coclustering,pnts1,clust_centres,Cov1,tempprobs)
        oracle_posterior    <-  update_oracle_list$oracle_posterior
        oracle_coclustering <- update_oracle_list$oracle_coclustering
      }
      else{
        for (i in 1:N) {
          for (j in 1:K) {
            oracle_posterior[j, i] <- tempprobs[j] * GPBayes::dmvnorm(pnts1[i,], clust_centres[j,], Cov1)
          }
          oracle_posterior[, i] <- oracle_posterior[, i] / sum(oracle_posterior[, i])
        }
        oracle_coclustering <- oracle_coclustering + t(oracle_posterior) %*% oracle_posterior
    for (c in 1:numiters) {
      tempprobs <- GPBayes::rdirichlet(1, rep(M, K))
      
      if(rcpp){
        update_oracle_list <- updateOracle_rcpp(oracle_posterior,oracle_coclustering,pnts2,clust_centres,Cov2,tempprobs)
        oracle_posterior <-  update_oracle_list$oracle_posterior
        oracle_coclustering <- update_oracle_list$oracle_coclustering
      }
      
      else{
        for (i in 1:N) {
          for (j in 1:K) {
            oracle_posterior[j, i] <- tempprobs[j] * GPBayes::dmvnorm(pnts2[i,], clust_centres[j,], Cov1)
          }
          oracle_posterior[, i] <- oracle_posterior[, i] / sum(oracle_posterior[, i])
        }
        oracle_coclustering <- oracle_coclustering + t(oracle_posterior) %*% oracle_posterior
      }
      
    }
    oracle_coclustering2 <- oracle_coclustering / numiters #oracle coclustering matrix
    
    
  }
  else{
    oracle_coclustering1 <- NULL
    oracle_coclustering2 <- NULL
  }
  
  
  return(list(points1 = pnts1, points2 = pnts2,
              probs = probs, 
              clusts1 = clusts1, clusts2 = clusts2,
              D1 = as.matrix(dist(pnts1)),
              D2 = as.matrix(dist(pnts2)),
              oracle_coclustering1 = oracle_coclustering1,
              oracle_coclustering2 = oracle_coclustering2))
}

#' Telescopic Dependence Between Clusterings
#'
#' Measures dependence between two clustering solutions.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Numeric value measuring dependence
#' @keywords internal
telescopic_dependence <- function(z1,z2,rcpp=TRUE){
  
  if(rcpp){
    return(telescopic_dependence_rcpp(z1,z2))
  }
  else{
    n <- length(z1)
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    sum = 0
    for(i in 1:(n-1)){
      for(j in (i+1):n){
        if(z2[i] == z2[j] & z1[i] == z1[j]) a1 <- a1 + 1 
        if(z1[i] == z1[j])                  a2 <- a2 + 1 
        if(z2[i] == z2[j] & z1[i] != z1[j]) a3 <- a3 + 1 
        if(z1[i] != z1[j])                  a4 <- a4 + 1 
        sum = sum + 1
      }
    }
    a1 <- a1/sum
    a2 <- a2/sum
    a3 <- a3/sum
    a4 <- a4/sum
    
    P1 <- a1/a2
    P2 <- a3/a4
  }
  return( (P1-P2)/P1 )
}

#' Convert Output List to Matrix
#'
#' Extracts named elements from a list of outputs and converts to matrix.
#'
#' @param output List of outputs from MCMC
#' @param output_name Name of element to extract
#'
#' @return Matrix of extracted values
#' @keywords internal
to_matrix <- function(output,output_name){
  if( length(output[[1]][[output_name]]) > 1 ){
    return(do.call(rbind,lapply(output, function(x) x[[output_name]])))
  }else{
    return(do.call(rbind,lapply(output, function(x) x[[output_name]]))[,1])
  }
}

######################## POST FITTING DIAGNOSTICS ###################################
#' Check Clustering Quality
#'
#' Computes clustering agreement metrics with a reference clustering.
#'
#' @param z.matrix Matrix of cluster assignments (rows are samples, columns are replications)
#' @param z.ref Reference cluster assignments
#' @param maxNClusters Maximum number of clusters to consider
#'
#' @return Data frame with clustering metrics (Binder, ARI, NVI, K)
#' @keywords internal
check_clustering <- function(z.matrix, z.ref, maxNClusters = 100){
  z <- salso::salso(z.matrix ,loss = "binder", maxNClusters)
  
  binder <- salso::binder(z.matrix, z.ref)
  ARI    <- salso::ARI(z.matrix, z.ref)
  NVI    <- salso::NVI(z.matrix, z.ref)
  K      <- length(unique(z))
  
  return(data.frame(binder=binder, ARI=ARI, NVI = NVI, K = K))
}


#' MCMC Convergence Diagnostics
#'
#' Computes Rhat and ESS diagnostics for MCMC samples.
#'
#' @param fit List of MCMC samples from mcmc_sampler
#'
#' @return Data frame with ESS and Rhat for each parameter
#' @keywords internal
convergence_diagnostics <- function(fit){
  
  alpha.d <- to_matrix(fit, "alpha.d") 
  M       <- to_matrix(fit, "M") 
  theta   <- to_matrix(fit, "theta")
  K1      <- apply(to_matrix(fit,"z1"),1,function(x)length(unique(x)))
  K2      <- apply(to_matrix(fit,"z2"),1,function(x)length(unique(x)))
  
  ess_alpha  <- posterior::ess_bulk(alpha.d)
  rhat_alpha <- posterior::rhat(alpha.d)
  
  output <- data.frame(variable = "alpha", ess_bulk = ess_alpha, rhat = rhat_alpha, acceptance_rate = NA)
  
  if(length(unique(M))>1){
    ess_M  <- posterior::ess_bulk(M)
    rhat_M <- posterior::rhat(M)
    output <- rbind(output,cbind(variable = "M", ess_bulk = ess_M, rhat = rhat_M, 
                                 acceptance_rate = length(unique(M))/length(M)))
  }
  
  if(length(unique(theta))>1){
    ess_theta  <- posterior::ess_bulk(theta)
    rhat_theta <- posterior::rhat(theta)
    output <- rbind(output,cbind(variable = "theta", ess_bulk = ess_theta, rhat = rhat_theta,
                                 acceptance_rate = length(unique(theta))/length(theta)))
  }  
  
  ess_K1  <- posterior::ess_bulk(K1)
  rhat_K1 <- posterior::rhat(K1)
  output  <- rbind(output,cbind(variable = "K1", ess_bulk = ess_K1, rhat = rhat_K1, acceptance_rate = NA))
  
  ess_K2  <- posterior::ess_bulk(K2)
  rhat_K2 <- posterior::rhat(K2)
  output  <- rbind(output,cbind(variable = "K2", ess_bulk = ess_K2, rhat = rhat_K2, acceptance_rate = NA))
  
  return(output)
}


#' Generate MCMC Diagnostic Plots
#'
#' Creates trace, histogram, and ACF plots for MCMC parameters.
#'
#' @param fit List of MCMC samples from mcmc_sampler
#' @param lags Number of lags for ACF plots
#'
#' @keywords internal
get_plots <- function(fit,lags = 50){
  alpha.d <- to_matrix(fit, "alpha.d") 
  M       <- to_matrix(fit, "M") 
  theta   <- to_matrix(fit, "theta")
  K1      <- apply(to_matrix(fit,"z1"),1,function(x)length(unique(x)))
  K2      <- apply(to_matrix(fit,"z2"),1,function(x)length(unique(x)))
  log_lik <- to_matrix(fit, "log_lik")
  
  posterior <- data.frame(alpha = alpha.d, M = M, theta = theta)
  print(bayesplot::mcmc_trace(posterior))
  print(bayesplot::mcmc_hist(posterior))
  print(bayesplot::mcmc_acf(posterior, lags = lags))
  
  posterior <- data.frame(log_lik = log_lik)
  print(bayesplot::mcmc_trace(posterior))
  print(bayesplot::mcmc_hist(posterior))
  print(bayesplot::mcmc_acf(posterior, lags = lags))
  
  posterior <- data.frame(K1 = K1, K2 = K2)
  print(bayesplot::mcmc_trace(posterior))
  print(bayesplot::mcmc_hist(posterior))
  print(bayesplot::mcmc_acf(posterior, lags = lags))
}


coclustering <- function(samples){
  n <- ncol(samples)
  s <- matrix(NA, n, n)
  for(i in 1:n){
    for(j in i:n){
      s[i,j] <- mean(samples[,i]==samples[,j])
      s[j,i] <- s[i,j]
    }
  }
  return(s) 
}
