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
get_log_prob_D_tesselation <- function(D, z, centers,  params){
  repulsion = FALSE
  linear   = FALSE
  if(!is.null(params[["repulsion"]])) repulsion = params[["repulsion"]]
  if(!is.null(params[["linear"]]))    linear    = params[["linear"]]
  if(linear){
    return(get_log_prob_D_tesselation_2_rcpp(D,z, centers, repulsion, params))
  }
  else{
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
tesselation <- function(D, centers){
  N                <- nrow(D)
  z                <- numeric(N)
  for(j in 1:N){
    if(!(j %in% centers)){
      z[j] = which.min(D[j,centers])
    }else{
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
get_probs <- function(type, D, gamma, centers, z, prior_param, tempering,  params, move_index = NULL){
  N         <- length(gamma)
  K         <- length(centers)
  
  
  log_probs <- rep(-Inf,N)
  prob <- rep(0,N)
  
  if(type == "birth"){
    
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
#' @keywords internal
mcmc_tesselation <- function(D, prior_param=3, params, N.sim = 10000, tempering = 0,
                             verbose = FALSE){
  
  
  params[["within_distance"]]  <- NULL
  params[["between_distance"]] <- NULL
  
  N                 <- nrow(D)
  centers           <- params$medoids.init
  gamma             <- rep(0, N)
  gamma[centers]    <- 1
  z                 <- tesselation(D, centers)
  log.mlik          <- get_log_prob_D_tesselation(D, z, centers,  params) 
  
  
  samples <- list()
  pb = txtProgressBar(min = 0, max = N.sim, initial = 0, style = 0) 
  start = proc.time()
  for(i in 1:N.sim){
    
    K <- length(centers)                        #number of centers
    #print(paste0("K = ",K))
    if(i %% 2 ==1 || K==0 || K == N){             #BIRTH OR DEATH STEP -------------------------
      birth = (runif(1)>0.5 || K < 2) & (K!=N) # prob > 0.5 or birth if no clusters and no cluster saturation -> if not then death!
      
      #print(birth)
      if(birth){
        new_probs <- get_probs("birth", D, gamma, centers, z, prior_param, tempering,  params)
      }else{
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
      #print("move")
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
    numerator   <- log.mlik.new  + log(prob_old_new) + log_prior_k(length(centers.new), N, prior_param)
    denominator <- log.mlik  + log(prob_new_old) + log_prior_k(length(centers),N,prior_param)
    acceptance_prob <-  exp( numerator - denominator ) 
    
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

###################################################################################
################## joint sampler ##################################################
###################################################################################
f_same <- function(pair) pair[1] == pair[2]
f_diff <- function(pair) pair[1] != pair[2]

rand_index <- function(z1, z2) {
  
  # Total number of pairs
  n <- length(z1)
  total_pairs <- n*(n-1)/2
  
  
  # Creating a matrix to compare each pair for both clusterings
  agree_same <-   combn(z1, 2, FUN = f_same) &   combn(z2, 2, FUN = f_same)
  agree_diff <-   combn(z1, 2, FUN = f_diff) &   combn(z2, 2, FUN = f_diff)
  
  # Summing agreements (both in the same cluster and in different clusters)
  TP_and_TN <- sum(agree_same) + sum(agree_diff)
  
  # Since each pair is counted twice, we need to divide by 2
  rand_index <- TP_and_TN / total_pairs
  
  return(rand_index)
}

#d = part_distance(medoids1, z2, D2)

#add condition if medoids1 = medoids2 then d=0
C_lik_log <- function(z1, z2, D2=NULL, a, b){
  #d       = 2/(salso::ARI(z2,z2.proj)+1)-1
  d        = 1/rand_index(z1,z2)-1
  
  return( lgamma(a) - lbeta(a,b) + log(GPBayes::HypergU(a, 1-b,d)))
}

##do better one where you have more alphas near 0 and 1 and less in the middle...
alpha_conditional_sample <- function(z1, z2, D2, a, b){
  
  #z2.proj = tesselation(D2, medoids1)
  #d       = 2/(salso::ARI(z2,z2.proj)+1)-1
  d        = 1/rand_index(z1,z2)-1
  
  xmin <- 1/10000  
  xmax <- 1-1/10000
  N    <- 1000
  alpha   <- seq( xmin, xmax,,N)
  phi     <- alpha/(1-alpha)
  log_pdf <-  -phi*d +(b-1)*log(1-alpha) +(a-1)*log(alpha) 
  pdf     <- unlogprob_to_prob(log_pdf)
  cdf     <- cumsum(pdf)
  quant   <- splinefun(x = cdf, y = alpha, ties = "mean")
  sample  <- quant(runif(1))
  if(sample<0) sample = xmin
  if(sample>1) sample = xmax
  return(sample)
}

#save D1, and D2 in fit attributes? maybe params?
get_samples_alpha <- function(fit, D2, a, b, N = NULL){
  
  if(is.null(N)){ N = length(fit)}
  
  check.seq <- round(seq(1,length(fit),,N)) #indeces to check
  
  alpha <- numeric(N)
  for(i in 1:N){
    alpha[i] <- alpha_conditional_sample(fit[[i]]$z1,  fit[[i]]$z2, D2, a, b)
  }
  return(alpha)
} 


generate_gamma_proposal <- function(i,  D, gamma, centers, z, prior_param, tempering,  params){
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
  
  else{                       #MOVE STEP ------------------------------------
    
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
  
  return(list(gamma.new    = gamma.new,
              centers.new  = centers.new,
              z.new        = z.new,
              log.mlik.new = log.mlik.new,
              prob_new_old = prob_new_old, 
              prob_old_new = prob_old_new))
}


mcmc_tesselation_joint  <- function(D1, D2, prior_param = 3, params,
                                    N.sim = 10000, tempering = 0,
                                    verbose = FALSE){
  
  params$hyperparam_lik_1[["within_distance"]]  <- NULL
  params$hyperparam_lik_1[["between_distance"]] <- NULL
  params$hyperparam_lik_2[["within_distance"]]  <- NULL
  params$hyperparam_lik_2[["between_distance"]] <- NULL
  
  N             <- nrow(D1)
  
  #variables for layer 1
  centers1           <- params$hyperparam_lik_1$medoids.init
  gamma1             <- rep(0, N)
  gamma1[centers1]   <- 1
  z1                 <- tesselation(D1, centers1)
  log.mlik1          <- get_log_prob_D_tesselation(D1, z1, centers1,  params$hyperparam_lik_1) 
  
  centers2           <- params$hyperparam_lik_2$medoids.init
  gamma2             <- rep(0, N)
  gamma2[centers2]   <- 1
  z2                 <- tesselation(D2, centers2)
  log.mlik2          <- get_log_prob_D_tesselation(D2, z2, centers2,  params$hyperparam_lik_2)
  
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
    
    ###accept/reject
    numerator   <- log.mlik1.new + log_prior_k(length(centers1.new), N, prior_param) + log(prob_old_new) + C_lik_log(z1.new, z2, D2, params$alpha.a, params$alpha.b) 
    denominator <- log.mlik1 + log_prior_k(length(centers1),N,prior_param)  + log(prob_new_old) + C_lik_log(z1, z2, D2, params$alpha.a, params$alpha.b)
    acceptance_prob <-  exp( numerator - denominator ) 
    
    
    
    if(runif(1) < acceptance_prob){
      z1       = z1.new
      centers1 = centers1.new
      gamma1   = gamma1.new
      log.mlik1 = log.mlik1.new 
    }
    
    
    ##Proposal for gamma2-------------------------------------------------------
    proposal     <- generate_gamma_proposal(i, D2, gamma2, centers2, z2, prior_param, tempering,  params$hyperparam_lik_2)
    gamma2.new   <- proposal$gamma
    prob_new_old <- proposal$prob_new_old
    prob_old_new <- proposal$prob_old_new
    centers2.new                <- which(gamma2.new == 1)
    z2.new                      <- tesselation(D2, centers2.new)
    log.mlik2.new               <- get_log_prob_D_tesselation(D2, z2.new, centers2.new,  params$hyperparam_lik_2)
    
    ###accept/reject
    numerator   <- log.mlik2.new + log_prior_k(length(centers2.new), N, prior_param) + log(prob_old_new)  + C_lik_log(z1, z2.new, D2, params$alpha.a, params$alpha.b) 
    denominator <- log.mlik2 + log_prior_k(length(centers2),N,prior_param)  + log(prob_new_old)  + C_lik_log(z1, z2, D2, params$alpha.a, params$alpha.b)
    acceptance_prob <-  exp( numerator - denominator ) 
    
    #
    #+  
    
    if(runif(1) < acceptance_prob){
      z2       = z2.new
      centers2 = centers2.new
      gamma2   = gamma2.new
      log.mlik2 = log.mlik2.new 
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


##########################################################################################
################## joint sampler nested ##################################################
##########################################################################################


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


marginal_prior_predictive_sampler_tesselation <- function(n, N.sim = 300, params, rcpp = TRUE){
  #I can generate samples from lambda, theta, alpha, rho.... 
  #how to generate samples from D?
  #simulate lambda_k
  list2env(params,.GlobalEnv) #REMOVE THIS!!!
  samples <- list()
  
  for(l in 1:N.sim){
    
    z <- params$z.init
    # Layer 0
    
    cluster_labels <- unique(z)
    n_clust = length(cluster_labels)
    
    cluster_labels <- unique(z)
    D <- matrix(NA, n, n)
    between_distances_1 <-  rgamma(n_clust*(n_clust-1)/2,params$delta2,params$theta)
    
    lambda  <- rgamma(n_clust, params$mu, params$beta)
    within_distances_1  <- c()
    for(i in 1:length(n_clust)){
      n_d_i = sum(z==i) 
      within_distances_1 <- c(within_distances_1, rgamma(n_d_i,params$delta1,lambda[which(z[i]==cluster_labels)]))
    }
    
    
    samples[[length(samples)+1]] <- list(within_distances_1 = within_distances_1,
                                         between_distances_1 = between_distances_1)
    
    
    if(l %% 100 == 0) print(paste0("iteration ", l))
  }
  
  return(samples)
  
}


