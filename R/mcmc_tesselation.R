#' Compute Log Likelihood for Tesselation
#'
#' Computes log probability of distance matrix for Voronoi tesselation.
#'
#' @param D Distance matrix
#' @param z Cluster allocation vector
#' @param centers Center indices
#' @param params Model parameters
#'
#' @return Log likelihood
#' @keywords internal
get_log_prob_D_tesselation <- function(D, z, centers, params){
  # print("mcmc_tesselation::get_log_prob_D_tesselation")
  # Parse linear parameter in params
  if(!is.null(params[["linear"]])) {
    linear = params[["linear"]]
  } else {
    stop("'linear' field in params is missing with no default.")
  }
  # Parse repulsion parameter in params  
  if(!is.null(params[["repulsion"]])) {
    repulsion = params[["repulsion"]]
  } else {
    stop("'repulsion' field in params is missing with no default.")
  }
  # Compute log probs accordingly
  if(linear) {
    return(get_log_prob_D_tesselation_2_rcpp(D,z, centers, repulsion, params))
  } else {
    return(get_log_prob_D_tesselation_rcpp(D,z, centers, repulsion, params))
  }
}


#' Compute Voronoi Tesselation
#'
#' Assigns each point to its nearest center (Voronoi assignment).
#'
#' @param D Distance matrix
#' @param centers Vector of center indices
#'
#' @return Numeric vector of cluster assignments
#' @keywords internal
tesselation <- function(D, centers) {
  N <- nrow(D)
  z <- numeric(N)
  for(j in 1:N) {
    # cat(sprintf("j: %g\n", j))
    if(!(j %in% centers)) {
      # print(D[j,centers])
      z[j] = which.min(D[j,centers])
    } else {
      z[j] = which(centers == j)
    }
  }
  return(z)
}


#' Compute Proposal Probabilities for Birth-Death-Move
#'
#' Computes probabilities for proposing changes to Voronoi centers.
#'
#' @param type Type of move: "birth", "death", or "move"
#' @param D Distance matrix
#' @param gamma Binary indicator of center status
#' @param centers Current center indices
#' @param z Current cluster assignments
#' @param prior_param Prior parameters
#' @param tempering Tempering parameter
#' @param params Model parameters
#' @param move_index Index for move operation
#'
#' @return Numeric vector of proposal probabilities
#' @keywords internal
get_probs <- function(type, D, gamma, centers, z, prior_param, tempering, params, move_index = NULL){
  # print("mcmc_tessellation::get_probs")
  N         <- length(gamma)
  K         <- length(centers)
  
  
  log_probs <- rep(-Inf,N)
  prob <- rep(0,N)
  
  if(type == "birth"){
    # print("i'm here")
    if(tempering == 0){
      prob[gamma==0] <- rep(1/(N-K),N-K)
      return(prob)
    } 
    
    indeces_to_flip <- which(gamma==0)
    for(k in indeces_to_flip){
      gamma.temp = gamma
      gamma.temp[k]      <- 1 
      centers.temp       <- which(gamma.temp == 1)
      
      z.temp = tesselation(D, centers.temp)
      
      log_probs[k] <- get_log_prob_D_tesselation(D, z.temp, centers.temp,  params) + log_prior_k(length(centers.temp),N,prior_param) 
    }
  }
  
  if(type == "death"){
    # print("I'm also here")
    if(tempering == 0){
      prob[gamma==1] <- rep(1/K,K)
      return(prob)
    } 
    
    indeces_to_flip <- which(gamma==1)
    for(k in indeces_to_flip){
      gamma.temp         <- gamma
      gamma.temp[k]      <- 0 
      centers.temp       <- which(gamma.temp == 1)
      
      z.temp = tesselation(D, centers.temp)
      
      log_probs[k] <- get_log_prob_D_tesselation(D, z.temp, centers.temp,  params) + log_prior_k(length(centers.temp),N,prior_param) 
    }
  }
  if(type == "move"){
    # print("or eventually here")
    if(tempering == 0){
      prob[gamma==0] <- rep(1/(N-K),N-K)
      return(prob)
    } 
    
    indeces_to_flip <- which(gamma==0)
    for(k in indeces_to_flip){
      gamma.temp             <- gamma
      gamma.temp[move_index] <- 0
      gamma.temp[k]          <- 1 
      centers.temp           <- which(gamma.temp == 1)
      
      z.temp = tesselation(D, centers.temp)
      
      log_probs[k] <- get_log_prob_D_tesselation(D, z.temp, centers.temp,  params) + log_prior_k(length(centers.temp),N,prior_param) 
    }
    
  }
  # print("out from the ifs")
  # print("log_probs:"); print(log_probs)
  if(tempering>0){
    log_probs <- tempering*log_probs
    return(unlogprob_to_prob(log_probs))
  } 
  else if(tempering == -1){
    probs <- unlogprob_to_prob(log_probs)
    probs <- probs/(1+probs)
    return(probs/sum(probs))
  }
}


#' Convert Log Probabilities to Normalized Probabilities
#'
#' Converts unnormalized log probabilities to normalized probability vector using log-sum-exp trick for numerical stability.
#'
#' @param log_prob Numeric vector of unnormalized log probabilities
#'
#' @return Numeric vector of normalized probabilities summing to 1
#' @keywords internal
unlogprob_to_prob <- function(log_prob){
  return(exp(log_prob-matrixStats::logSumExp(log_prob)))
}


#' Log Prior for Number of Clusters
#'
#' Computes log prior probability for a given number of clusters.
#'
#' @param k Number of clusters
#' @param N Maximum number of clusters (data size)
#' @param prior_param Prior parameter(s). If scalar, uses geometric distribution; otherwise uses provided probability vector
#'
#' @return Log prior probability
#' @keywords internal
log_prior_k <- function(k, N, prior_param){
  if(length(prior_param)==1)  return(-lchoose(N,k) + dgeom(k,1/(prior_param+1),log=TRUE))
  else(return(log(prior_param[k])))
}


#' MCMC Sampler for Voronoi Tesselation with Birth-Death-Move
#'
#' Performs MCMC sampling over Voronoi tesselation configurations using birth-death-move updates.
#' 
#' @param D Distance matrix (should be upper triangular with diagonal of zeros)
#' @param prior_param Prior parameter for number of clusters (default 3)
#' @param params List of model parameters including medoids.init
#' @param N.sim Number of MCMC iterations (default 10000)
#' @param tempering Tempering parameter (default 0)
#' @param verbose Logical; print iteration details (default FALSE)
#'
#' @return List of MCMC samples
#' @export
mcmc_tesselation <- function(D, prior_param = 3, params, N.sim = 10000, tempering = 0,
                             verbose = FALSE){
  
  # Remove exact zeros from D
  D[which(D == 0, arr.ind = F)] <- 1e-6
  
  params[["within_distance"]]  <- NULL
  params[["between_distance"]] <- NULL
  
  N                 <- nrow(D)
  centers           <- params$medoids.init
  gamma             <- rep(0, N)
  gamma[centers]    <- 1
  z                 <- tesselation(D, centers)
  log.mlik          <- get_log_prob_D_tesselation(D, z, centers, params)
  # cat(sprintf("log.mlik: %g\n", log.mlik))
  
  
  samples <- list()
  cat(sprintf("  VoronoiClust: Tessellation MCMC (%d iterations)\n", N.sim))
  pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 3, width = 50, char = "*") 
  start = proc.time()
  for(i in 1:N.sim){
    
    K <- length(centers)                        #number of centers
    #print(paste0("K = ",K))
    if(i %% 2 ==1 || K==0 || K == N){             #BIRTH OR DEATH STEP -------------------------
      birth = (runif(1)>0.5 || K < 2) & (K!=N) # prob > 0.5 or birth if no clusters and no cluster saturation -> if not then death!
      
      #print(birth)
      if(birth){
        # print("birth")
        new_probs <- get_probs("birth", D, gamma, centers, z, prior_param, tempering,  params)
      }else{
        # print("death")
        new_probs <- get_probs("death", D, gamma, centers, z, prior_param, tempering,  params)
      }
      
      #print(birth)
      
      flip.index                 <- sample(1:N, 1, prob = new_probs)         #informed proposal...
      
      gamma.new                  <- gamma
      gamma.new[flip.index]      <- 1 - gamma.new[flip.index]
      centers.new                <- which(gamma.new == 1)
      z.new                      <- tesselation(D, centers.new)
      #print(centers.new)
      log.mlik.new               <- get_log_prob_D_tesselation(D, z.new, centers.new,  params)
      
      #get transition probs
      if(birth){
        prob_old_new <- get_probs("death", D, gamma.new, centers.new, z.new, prior_param, tempering,  params)[flip.index]
      }else{
        prob_old_new <- get_probs("birth", D, gamma.new, centers.new, z.new, prior_param, tempering,  params)[flip.index]
      }
      
      prob_new_old   <- new_probs[flip.index]
      
      
    }
    
    else{                       #MOVE STEP ------------------------------------
      # print("move")
      move_index     <- sample(centers, 1)
      new_probs  <- get_probs("move", D,  gamma, centers, z, prior_param, tempering,  params, move_index = move_index) #pick one at random and choose 
      
      flip.index                 <- sample(1:N, 1, prob = new_probs)         #informed proposal...
      gamma.new                  <- gamma
      gamma.new[move_index]      <- 0
      gamma.new[flip.index]      <- 1
      centers.new                <- which(gamma.new == 1)
      z.new                      <- tesselation(D, centers.new)
      #print(centers.new)
      
      log.mlik.new               <- get_log_prob_D_tesselation(D, z.new, centers.new,  params)
      
      #get transition probabilities
      prob_old_new <- get_probs("move", D,  gamma.new, centers.new, z.new, prior_param, tempering,  params, move_index = flip.index)[move_index]
      prob_new_old   <- new_probs[flip.index]
    }
    
    
    ##accept reject step...
    # print("MH step computation")
    numerator   <- log.mlik.new  + log(prob_old_new) + log_prior_k(length(centers.new), N, prior_param)
    # cat(sprintf("numerator: %g\n", numerator))
    denominator <- log.mlik  + log(prob_new_old) + log_prior_k(length(centers),N,prior_param)
    # cat(sprintf("denominator: %g\n", denominator))
    acceptance_prob <-  exp( numerator - denominator )
    # cat(sprintf("acceptance_prob: %g", acceptance_prob))
    
    utils::setTxtProgressBar(pb,i)
    if(runif(1) < acceptance_prob){
      z       = z.new
      centers = centers.new
      gamma   = gamma.new
      log.mlik = log.mlik.new 
      
      if(verbose){
        print(paste0("iteration ", i))
        #print(z)
        #print(i)
        print(centers)
      }
      
      
    }
    
    
    samples[[length(samples)+1]] <- list(z=z, 
                                         K = length(unique(z)), 
                                         centers = centers,
                                         log_lik = log.mlik,
                                         time = proc.time() - start)
    
  }
  close(pb)
  
  return(samples)
}


##########################################################################################
################## joint sampler nested ##################################################
##########################################################################################


#' Nested Voronoi Tesselation
#'
#' Computes a restricted Voronoi tesselation for z2 that respects the partition structure of z1.
#'
#' @param D2 Distance matrix for layer 2
#' @param centers2 Center indices for layer 2
#' @param z1 Clustering for layer 1 (constraint)
#'
#' @return Numeric vector of cluster assignments for z2
#' @keywords internal
nested_tesselation <- function(D2, centers2, z1){
  N                <- nrow(D2)
  z2               <- numeric(N)
  ##compute restricted tessellation 
  cluster_labels_1 <- unique(z1)
  for (k in cluster_labels_1){
    R_k            <- which(z1==k)
    if(length(R_k)==1){
      z2[R_k] = which(centers2 == R_k)
    }
    centers_in_R_K <- intersect(R_k, centers2)     
    
    for(j in R_k){
      if(length(centers_in_R_K)==0){    #no centers2 in R_k -> just consider singletons!
        z2[j] = 0                      #ATTENTION HERE!!!!!!!!!!!!!!!!!!!!!!
        #z2[j]   <- which.min(D2[j,centers2])
      }else if(length(centers_in_R_K)==1){
        z2[j] = which(centers2 == centers_in_R_K)
      }else{
        if(!(j %in% centers2)){
          min_index_in_R_K <- which.min(D2[j, centers_in_R_K]) #if equals to 2 than smallest distance is with second center in R_k
          #z2.new[j]        <- which(centers2.new == centers_in_R_K[min_index_in_R_K]) #link R_k center index with global center index
          z2[j]   <- which(centers2 == centers_in_R_K[min_index_in_R_K])
        }else{
          z2[j] = which(centers2 == j)
        }
        
      }
    }
  }
  z2[z2==0] <- (max(z2)+1):(max(z2)+sum(z2==0))
  
  return(z2)
}


#' Log Prior for Compatible Nested Gamma
#'
#' Computes log prior probability for nested clustering configuration that respects layer 1 partition.
#'
#' @param z1.new Clustering for layer 1
#' @param centers2.new Center indices for layer 2
#'
#' @return Log prior probability
#' @keywords internal
log_prior_gamma2_gamma1_2 <- function(z1.new, centers2.new){
  
  log_sum <- 0
  
  cluster_labels_1 <- unique(z1.new)
  for (k in cluster_labels_1){
    R_k  <- which(z1.new==k)
    cluster_size <- length(R_k)
    if(cluster_size==1) next
    number_of_centers_k <- length(intersect(centers2.new, R_k)) 
    if(number_of_centers_k == 0) return(-Inf)
    log_sum = log_sum - lchoose(cluster_size, number_of_centers_k) - log(cluster_size)  
  }
  
  return(log_sum)
}


#' Compute Proposal Probabilities for Nested Voronoi
#'
#' Computes probabilities for proposing changes to centers within nested partition structure.
#'
#' @param D Distance matrix
#' @param gamma Binary indicator of center status
#' @param z Cluster assignments
#' @param prior_k_mean Prior parameter for number of clusters
#' @param tempering Tempering parameter
#' @param update_group Indices to consider for updates (NULL = all)
#' @param z1.ref Reference clustering to maintain partition structure
#'
#' @return Numeric vector of proposal probabilities
#' @keywords internal
get_probs_nested <- function(D,gamma,z, prior_k_mean, tempering, update_group = NULL, z1.ref){
  N         <- length(gamma)
  log_probs <- numeric(N)
  
  if(is.null(update_group)){update_group = 1:N}
  
  if(tempering == 0) return(rep(1/length(update_group),length(update_group)))
  
  
  for(k in update_group){
    gamma.temp = gamma
    z.temp     = z
    gamma.temp[k]      <- 1 - gamma.temp[k]
    centers.temp       <- which(gamma.temp == 1)
    
    if(sum(gamma.temp[update_group])==0){           ##no centers
      gamma.temp[update_group] <- 1 
      print("warning: there are 0 centres")
      centers.temp       <- which(gamma.temp == 1)
      z.temp = nested_tesselation(D2 = D, centers2 = centers.temp, z1 = z1.ref)
      log_probs[k] <- get_log_prob_D_tesselation(D, z.temp, params$hyperparam_lik_1) + log_prior_k(length(centers.temp),N,prior_k_mean) 
      
    }else{
      z.temp = nested_tesselation(D2 = D, centers2 = centers.temp, z1 = z1.ref)
      
      log_probs[k] <- get_log_prob_D_tesselation(D, z.temp, params$hyperparam_lik_1) + log_prior_k(length(centers.temp),N,prior_k_mean) 
    }
    
  }
  
  log_probs = log_probs[update_group]
  
  if(tempering>0){
    log_probs <- tempering*log_probs
    return(unlogprob_to_prob(log_probs))
  } 
  else if(tempering == -1){
    probs <- unlogprob_to_prob(log_probs)
    probs <- probs/(1+probs)
    return(probs/sum(probs))
  }
}


#' Nested MCMC Sampler for Two-Layer Voronoi Tesselation
#'
#' Performs MCMC sampling with nested Voronoi tesselation where z2 respects the partition structure of z1.
#'
#' @param D1 First distance matrix
#' @param D2 Second distance matrix
#' @param prior_param Prior parameter for number of clusters (default 3)
#' @param params List of model parameters
#' @param N.sim Number of MCMC iterations (default 10000)
#' @param tempering Tempering parameter (default 0)
#' @param repulsion Logical; include repulsion parameter (default TRUE, unused)
#' @param verbose Logical; print iteration details (default FALSE)
#'
#' @return List of MCMC samples with nested structure
#' @keywords internal
mcmc_tesselation_nested <- function(D1, D2, prior_param = 3, params,
                                    N.sim = 10000, tempering = 0, repulsion = TRUE,
                                    verbose = FALSE){
  N <- nrow(D1)
  
  params$hyperparam_lik_1[["within_distance"]]  <- NULL
  params$hyperparam_lik_1[["between_distance"]] <- NULL
  params$hyperparam_lik_2[["within_distance"]]  <- NULL
  params$hyperparam_lik_2[["between_distance"]] <- NULL
  
  #variables for layer 1
  centers1           <- params$hyperparam_lik_1$medoids.init
  gamma1             <- rep(0, N)
  gamma1[centers1]   <- 1
  z1                 <- tesselation(D1, centers1)
  log.mlik1          <- get_log_prob_D_tesselation(D1, z1, centers1,  params$hyperparam_lik_1) 
  
  
  gamma2              <- gamma1
  centers2            <- which(gamma2==1)
  #print("pos1")
  z2                  <- nested_tesselation(D2, centers2, z1)
  log.mlik2           <- get_log_prob_D_tesselation(D2, z2, centers2,  params$hyperparam_lik_2) 
  
  
  samples <- list()
  pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 0) 
  start = proc.time()
  for(i in 1:N.sim){
    
    ##Proposal for gamma1-------------------------------------------------------
    proposal       <- generate_gamma_proposal(i, D1, gamma1, centers1, z1, prior_param, tempering,  params$hyperparam_lik_1)
    gamma1.new     <- proposal$gamma.new
    centers1.new   <- proposal$centers.new
    z1.new         <- proposal$z.new
    log.mlik1.new  <- proposal$log.mlik.new
    prob_new_old   <- proposal$prob_new_old
    prob_old_new   <- proposal$prob_old_new
    
    ##compatible proposal for gamma2-------------------------------------------
    cluster_labels_1 <- unique(z1.new)
    gamma2.new      <- gamma2
    for (k in cluster_labels_1){
      R_k  <- which(z1.new==k)
      if(length(R_k)==1){gamma2.new[R_k] = 1} #I can remove this it is implied next
      else if(sum(gamma2.new[R_k])==0){
        gamma2.new[R_k] = gamma1.new[R_k]
      }
    }
    centers2.new        <- which(gamma2.new==1)
    #print("pos2")
    z2.new              <- nested_tesselation(D2, centers2.new, z1.new)
    log.mlik2.new       <- get_log_prob_D_tesselation(D2, z2.new, centers2.new,  params$hyperparam_lik_2)
    
    ###accept/reject
    numerator   <- log.mlik1.new + log.mlik2.new + log_prior_k(length(centers1.new), N, prior_param)  + log_prior_k(length(centers2.new ), N, prior_param) + log(prob_old_new) 
    denominator <- log.mlik1 + log.mlik2 + log_prior_k(length(centers1),N,prior_param) + log_prior_k(length(centers2),N,prior_param)  + log(prob_new_old) 
    acceptance_prob <-  exp( numerator - denominator ) 
    
    
    if(runif(1)< acceptance_prob){
      z1       = z1.new
      centers1 = centers1.new
      gamma1   = gamma1.new
      log.mlik1= log.mlik1.new
      
      z2        = z2.new
      centers2  = centers2.new
      gamma2    = gamma2.new
      log.mlik2 = log.mlik2.new 
    }
    
    
    ########### second part - for each subpartitio do accept reject 10 times? adjust later
    m2=1
    cluster_labels_1 <- unique(z1)
    gamma2.new <- gamma2
    for (k in cluster_labels_1){ #cycle trough all clusters in layer 1
      #print(k)
      #print(R_k)
      R_k  <- which(z1==k)
      if(length(R_k)>1){
        
        for(k2 in 1:m2){
          new_probs                  <- get_probs_nested(D2,gamma2,z2, prior_k_mean, tempering, update_group = R_k, z1.ref = z1)    #informed proposal...
          #print(R_k)
          #print(new_probs)
          choose.index               <- sample(1:length(R_k), 1, prob = new_probs)
          flip.index                 <- R_k[choose.index]                                     #informed proposal...
          z2.new                     <- z2
          gamma2.new                 <- gamma2
          gamma2.new[flip.index]     <- 1 - gamma2.new[flip.index]
          centers2.new               <- which(gamma2.new == 1)
          #print("pos3")
          z2.new                     <- nested_tesselation(D2, centers2.new, z1)
          
          log.mlik.new2              <- get_log_prob_D_tesselation(D2, z2.new, centers2.new,  params$hyperparam_lik_2) 
          
          numerator                  <- log.mlik.new2 + log_prior_gamma2_gamma1_2(z1, centers2.new) + get_probs_nested(D2,gamma2.new,z2.new, prior_k_mean, tempering, update_group = R_k, z1.ref = z1)[choose.index]
          denominator                <- log.mlik2     + log_prior_gamma2_gamma1_2(z1,  centers2)    + new_probs[choose.index] 
          acceptance_prob            <- min(1, exp( numerator - denominator ) )
          
          if(runif(1)< acceptance_prob){
            z2        = z2.new
            centers2  = centers2.new
            gamma2    = gamma2.new
            log.mlik2 = log.mlik.new2 
          }
        }
        
      }
    }
    
    setTxtProgressBar(pb,i)
    if(verbose){
      print(paste0("iteration ", i))
      print(centers1)
      print(centers2)
    }
    
    samples[[length(samples)+1]] <- list(z1=z1, z2 = z2, 
                                         K1 = length(unique(z1)), K2 = length(unique(z2)), 
                                         centers1 = centers1, centers2 = centers2,
                                         log_lik = log.mlik1 + log.mlik2,
                                         time = proc.time() - start)
    
  }
  close(pb)
  
  
  return(samples)
}


#################################################
##prior predictive sampler#######################
#################################################


#' Marginal Prior Predictive Sampler for Tesselation
#'
#' Generates samples from the prior predictive distribution of distances under tesselation model.
#'
#' @param n Number of observations
#' @param N.sim Number of samples to generate (default 300)
#' @param params Model parameters
#' @param rcpp Logical; use Rcpp implementation (default TRUE, currently unused)
#'
#' @return List of samples with within and between distance distributions
#' @export
# marginal_prior_predictive_sampler_tesselation <- function(n, N.sim = 300, params, rcpp = TRUE){
#   #I can generate samples from lambda, theta, alpha, rho.... 
#   #how to generate samples from D?
#   #simulate lambda_k
#   # list2env(params,.GlobalEnv) #REMOVE THIS!!!
#   samples <- list()
#   
#   for(l in 1:N.sim){
#     
#     z <- params$z.init
#     # Layer 0
#     
#     cluster_labels <- unique(z)
#     n_clust = length(cluster_labels)
#     
#     cluster_labels <- unique(z)
#     D <- matrix(NA, n, n)
#     between_distances_1 <-  rgamma(n_clust*(n_clust-1)/2, params$delta2, params$theta)
#     
#     lambda  <- rgamma(n_clust, params$mu, params$beta)
#     within_distances_1  <- c()
#     for(i in 1:length(n_clust)){
#       n_d_i = sum(z==i) 
#       within_distances_1 <- c(within_distances_1, rgamma(n_d_i,params$delta1,lambda[which(z[i]==cluster_labels)]))
#     }
#     
#     
#     samples[[length(samples)+1]] <- list(within_distances_1 = within_distances_1,
#                                          between_distances_1 = between_distances_1)
#     
#     
#     if(l %% 100 == 0) print(paste0("iteration ", l))
#   }
#   
#   return(samples)
# }
marginal_prior_predictive_sampler_tesselation <- function(n, N.sim = 300, params) {

  samples <- list()

  for(l in 1:N.sim){

    z <- params$z.init
    cluster_labels <- unique(z)
    n_clust = length(cluster_labels)

    # D <- matrix(NA, n, n)

    # Sample latent between clusters parameters (if quadratic)
    theta <- matrix(0, n_clust, n_clust)
    if(params$linear == TRUE) {
      theta[upper.tri(theta)] <- params$theta
    } else {
      theta[upper.tri(theta)] <- rgamma(n_clust*(n_clust-1)/2, params$zeta, params$gamma)
    }

    # Sample between distances
    between_distances_1 <- c()
    for (k in 1:(n_clust-1)) {
      for(t in (k+1):n_clust) {
        nk <- sum(z == k); nt <- sum(z == t)
        between_distances_1 <- c(between_distances_1, rgamma(nk*nt, params$delta2, theta[k,t]))
      }
    }
    # between_distances_1 <-  rgamma(n_clust*(n_clust-1)/2, params$delta2, params$theta)

    # Sample latent within cluster parameters
    lambda  <- rgamma(n_clust, params$mu, params$beta)

    # Samplewithin distances
    within_distances_1  <- c()
    for(i in 1:n_clust){
      ni = sum(z == i)
      within_distances_1 <- c(within_distances_1, rgamma(0.5*ni*(ni-1), params$delta1, lambda[i]))
    }

    # Append samples
    samples[[length(samples)+1]] <- list(within_distances_1 = within_distances_1,
                                         between_distances_1 = between_distances_1)

    # if(l %% 100 == 0) print(paste0("iteration ", l))
  }

  return(samples)
}



