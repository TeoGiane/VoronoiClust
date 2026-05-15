#' Helper Function for Cluster Elements
#'
#' Returns indices of elements in cluster k, excluding element i.
#'
#' @param z Cluster allocation vector
#' @param k Cluster index
#' @param i Element index to exclude
#'
#' @return Numeric vector of element indices
#' @keywords internal
elements_in_cluster <- function(z, k, i){
  elements_in_k <- which(z==k)
  return(elements_in_k[elements_in_k!=i])
}

#' Compute Log Likelihood of Data Given Clustering
#'
#' Computes log probability of distance matrix given cluster allocations.
#'
#' @param D Distance matrix
#' @param z Cluster allocation vector
#' @param params List of model parameters
#' @param rcpp Logical; use Rcpp implementation
#' @param return_L_matrix Logical; if TRUE return full likelihood matrix
#'
#' @return Log likelihood or list with log likelihood and matrix
#' @keywords internal
get_log_prob_D <- function(D,z,params, rcpp = TRUE, return_L_matrix = FALSE){

  
  if(rcpp){
    
    if(return_L_matrix){
      return(get_log_prob_D_rcpp_list(D,z,params))
    }
    else{
      return(get_log_prob_D_rcpp(D,z,params))
    }
  }  
  
  else{
    list2env(params,.GlobalEnv)
    unique_z   <- sort(unique(z))
    K          <- length(unique_z)
    n          <- length(z)
    
    #check if clusters are 11 22 33 and not 11 22 44 -> if this gives error than stop...
    if(any(diff(sort(unique(z)))>1)) warning("Cluster allocations have wrong format")
      
    log_Lkt <- matrix(0,K,K)
    
    for(k in 1:K){
      index_select = (1:n)[z==unique_z[k]]     #indeces of elements in cluster k
      D_k <- D[index_select, index_select]  #Distance matrix for this cluster elements
      distances <- na.omit(D_k[upper.tri(D_k)])
      log_Lkt[k,k] <- mu*log(beta) + (delta1-1)*sum(log(distances)) + (-mu-delta1*length(distances))*log(beta+sum(distances))  -length(distances)*lgamma(delta1) + lgamma(mu+length(distances)*delta1) - lgamma(mu)
    }
    
    if(K>1){
      for(k in 1:(K-1)){
        for(t in (k+1):K){
          index_select1 = (1:n)[z==unique_z[k]]      #indeces of elements in cluster k
          index_select2 = (1:n)[z==unique_z[t]]      #indeces of elements in cluster t
          
          D_kt <- D[index_select1, index_select2] 
          distances <- na.omit(c(D_kt))
          log_Lkt[k,t] <- zeta*log(gamma) + (delta2-1)*sum(log(distances)) + (-zeta-delta2*length(distances))*log(gamma+sum(distances))  -length(distances)*lgamma(delta2) + lgamma(zeta+length(distances)*delta2) - lgamma(zeta)
        }
      }
    }
    
    log_lik = sum(log_Lkt, na.rm = TRUE) #repulsion part seems to have huge effect (check this latter)
    
    if(return_L_matrix){
      return(list(log_lik = log_lik, log_Lkt = log_Lkt))
    }
    else{
      return(log_lik)
    }
  }
 
}

#' Subordinator Function
#'
#' Computes the log of a subordinator used in PY process.
#'
#' @param x Numeric value
#' @param m Number of terms
#' @param alpha Discount parameter
#'
#' @return Log of subordinator
#' @keywords internal
log_subor <- function(x, m, alpha=1){
  if(m==0) return(log(1))
  if(m==1) return(log(x))
  if(m>1){
    log_prob = log(x)
    for(i in 1:(m-1)){
      log_prob = log_prob + log(x+(i)*alpha)
    }
  }
  return(log_prob)
}

#' Compute Log Probability of Partition
#'
#' Computes log probability of a partition under Pitman-Yor process prior.
#'
#' @param z Cluster allocation vector
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Log probability of partition
#' @keywords internal
get_log_prob_partition <- function(z, M, theta, rcpp = TRUE){
 
  if(rcpp){
    return(get_log_prob_partition_rcpp(z, M, theta))
  }  
  else{
    if(length(z)==0) return(0)
    K          <- length(unique(z))
    nk         <- plyr::count(z)[,2]
    n          <- length(z)
    
    log_prob <- log_subor(M+theta,K-1,theta) - log_subor(M+1,n-1) 
    for( i in 1:K){
      log_prob <- log_prob + log_subor(1-theta,nk[i]-1)
    }
    
    return(log_prob) 
  }

}

#check if makes sense
#exp(get_log_prob_partition(c(1,1,1),5,0.5))+exp(get_log_prob_partition(c(1,2,3),5,0.5))+3*exp(get_log_prob_partition(c(1,2,2),5,0.5))
#https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/pitman95a.pdf

#' Compute Log Cluster Allocation Probabilities
#'
#' Computes log probabilities for assigning element i to each cluster.
#'
#' @param z Cluster allocation vector
#' @param i Element index
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param D Distance matrix
#' @param params Model parameters
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Numeric vector of log probabilities
#' @keywords internal
get_log_prob_allocations <- function(z, i, M, theta, D, params, rcpp = TRUE){
  if(rcpp){
    return(get_log_prob_allocations_rcpp(z, i, M, theta, D, params))
  }  
  else{
    list2env(params,.GlobalEnv)
    K_mi       <- length(unique(z[-i]))
    log_probs  <- numeric(K_mi+1) 
    n          <- length(z)
    
    
    #probability of being assigned to existing clusters
    for(k in 1:(length(log_probs))){
      z_temp    <- z
      z_temp[i] <- k
      log_probs[k] <- get_log_prob_partition(z_temp,M,theta,rcpp) + get_log_prob_D(D, z_temp, params$hyperparam_lik_2,rcpp)
    }
    
    #print(get_log_prob_partition(z_temp,M,theta))
    #print( get_log_prob_D(D,z_temp,params))
    return(log_probs)
  }
}

#' Compute Log Allocation Probabilities with Compatibility Check
#'
#' Computes log probabilities with constraint that z1 and z2 have same partition where gamma=1.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param gamma.d Binary indicator vector
#' @param i Element index
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param D Distance matrix
#' @param params Model parameters
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Numeric vector of log probabilities
#' @keywords internal
get_log_prob_allocations_comp_check<- function(z1, z2, gamma.d, i, M, theta, D, params, rcpp=TRUE){
  if(rcpp){
    return(get_log_prob_allocations_comp_check_rcpp(z1, z2, gamma.d, i, M, theta, D, params))
  }  
  else{
    
    list2env(params,.GlobalEnv)
    K_mi  <- length(unique(z1[-i]))
    log_probs <- numeric(K_mi+1) 
    
    #probability of being assigned to existing clusters
    for(k in 1:(length(log_probs))){
      z1_temp    = z1
      z1_temp[i] = k 
      compatibility = check_same_partition(z1_temp[gamma.d==1], z2[gamma.d==1])
      if(!compatibility){
        log_probs[k]=-Inf
        #  print("Compatibility check 2 -- delete this")
      }
      else{
        log_probs[k] <- get_log_prob_partition(z1_temp,M,theta,rcpp) + get_log_prob_D(D, z1_temp,  params$hyperparam_lik_1,rcpp)
      }
      
    }
    return(log_probs)
    
  }
}

#' Update Gamma Indicators
#'
#' Updates binary indicators that determine shared structure between two clusterings.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param gamma.d Binary indicator vector
#' @param alpha.d Concentration parameter
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated gamma.d vector
#' @keywords internal
update_gammas <- function(z1, z2, M, theta, gamma.d, alpha.d,rcpp=TRUE){
  if(rcpp){
    return(c(update_gammas_rcpp(z1, z2, M, theta, gamma.d, alpha.d)))
  }  
  else{
    n <- length(z1)
    for(i in 1:length(z1)){
      old_gammai = gamma.d[i]
      z2_R_pi <- z2[ (gamma.d==1) | ((1:n)==i)]
      z2_R_mi <- z2[ (gamma.d==1 & (1:n)!=i)]
      
      
      if(length(z2_R_mi)==0){ #then all gammas are 0 and posterior of gamma does not depend direclty on p1 or p2
        ratio =1}
      else{
        ratio = exp(get_log_prob_partition(z2_R_pi, M,theta,rcpp) - get_log_prob_partition(z2_R_mi, M,theta,rcpp))
      }
      
      prob = alpha.d/(alpha.d + (1-alpha.d)*ratio)
      #print(prob)
      gamma.d[i] = rbinom(1,1,prob)
       
       if( (old_gammai == 0) & (gamma.d[i] == 1)){
         compatibility <- check_same_partition(z1[gamma.d==1| (1:n)==i], z2[gamma.d==1| (1:n)==i])
         
         if(!compatibility){
           gamma.d[i] = 0
           #print("not compatible - remove this comment")
         }
       }
    }  
    return(gamma.d)
  }
}



#' Update First Clustering Vector
#'
#' Updates cluster allocations in z1 with compatibility constraints from z2 and gamma.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param gamma.d Binary indicator vector
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param D1 First distance matrix
#' @param params Model parameters
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated z1 vector
#' @keywords internal
update_z1 <- function(z1,z2,gamma.d,M,theta,D1,params, rcpp){
  if(rcpp){
    return(c(update_z1_rcpp(z1,z2,gamma.d,M,theta,D1,params)))
  }  
  else{
    for(j in 1:n){
      log_probs <- get_log_prob_allocations_comp_check(z1,z2,gamma.d,j,M,theta,D1,params, rcpp)
      z1        <- update_z(z1,j,log_probs,rcpp)
    }
  }
  return(z1)
}

#' Update Second Clustering Vector
#'
#' Updates cluster allocations in z2, only for observations where gamma=0.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param gamma.d Binary indicator vector
#' @param M Mass parameter
#' @param theta Discount parameter
#' @param D2 Second distance matrix
#' @param params Model parameters
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated z2 vector
#' @keywords internal
update_z2 <- function(z1,z2,gamma.d,M,theta,D2,params, rcpp){
  if(rcpp){
    return(c(update_z2_rcpp(z1,z2,gamma.d,M,theta,D2,params)))
  }  
  else{
    update_indeces <- (1:n)[gamma.d==0]
    for(j in update_indeces){
      log_probs <- get_log_prob_allocations(z2,j,M,theta,D2,params,rcpp)
      z2        <- update_z(z2,j,log_probs,rcpp)
    }
    return(z2)
  }
  
}


#' Convert Log Probabilities to Normalized Probabilities
#'
#' Converts unormalized log probabilities to normalized probability vector.
#'
#' @param log_prob Numeric vector of log probabilities
#'
#' @return Numeric vector of normalized probabilities
#' @keywords internal
unlogprob_to_prob <- function(log_prob){
  return(exp(log_prob-matrixStats::logSumExp(log_prob)))
}


#' Check if Two Clusterings Represent Same Partition
#'
#' Compares two clustering vectors to determine if they represent the same partition.
#'
#' @param c1 First clustering vector
#' @param c2 Second clustering vector
#'
#' @return Logical; TRUE if partitions are identical
#' @keywords internal
check_same_partition <- function(c1,c2){
  if(length(c1)!=length(c2)) return(FALSE)
  if(length(c1)==0) return(TRUE)
  
  same = igraph::compare(c(-1,-1,c1),c(-1,-1,c2)) == 0
  
  return(same)
}


#' Update Single Cluster Assignment
#'
#' Updates cluster assignment for element j using Gibbs sampling with cluster relabeling for singletons.
#'
#' @param z Clustering vector
#' @param j Index of element to update
#' @param log_probs Numeric vector of log probabilities for each cluster
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated clustering vector
#' @keywords internal
update_z <- function(z,j,log_probs,rcpp=TRUE){
  #check if z[i] belongs to a single cluster
  singleton = sum(z==z[j])==1
  
  #if z[i] is a singleton and is assigned a new label then shift all elements above the last allocation by minus 1
  if(singleton){
    last_allocation = z[j]
    z[j] <- sample(1:length(log_probs), size = 1, prob = unlogprob_to_prob(log_probs))
    if(z[j] != last_allocation){
      z[z>last_allocation] = z[z>last_allocation]-1 
    }
  }else{
    z[j] <- sample(1:length(log_probs), size = 1, prob = unlogprob_to_prob(log_probs))
  }
  
  return(z)
}


#' Update M Parameter in Joint MCMC Sampler
#'
#' Performs Metropolis-Hastings step to update the M parameter in joint sampler.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param M Current value of M parameter
#' @param theta Current value of theta parameter
#' @param gamma.d Binary indicator vector
#' @param params Model parameters including metropolis_M_sd and lambda.M
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated value of M parameter
#' @keywords internal
update_M <- function(z1, z2, M, theta, gamma.d, params,rcpp){
  list2env(params,.GlobalEnv)
  
  candidate_M   <- truncnorm::rtruncnorm(1, a=0, mean = M, sd = metropolis_M_sd)
  
  z2_R <- z2[(gamma.d==1)]
  
  candidate_log_prob <-  get_log_prob_partition(z2, candidate_M, theta,rcpp) + get_log_prob_partition(z1, candidate_M, theta,rcpp) - get_log_prob_partition(z2_R, candidate_M, theta,rcpp) + dexp(candidate_M,lambda.M, log = TRUE)
  old_log_prob       <-  get_log_prob_partition(z2, M, theta,rcpp) + get_log_prob_partition(z1, M, theta,rcpp) - get_log_prob_partition(z2_R, M, theta,rcpp) + dexp(M, lambda.M, log = TRUE)

  log_transition_prob_old_new <- log(dtruncnorm(M, mean = candidate_M, sd = metropolis_M_sd))
  log_transition_prob_new_old <- log(dtruncnorm(candidate_M, mean = M, sd = metropolis_M_sd))
  
  accept_ratio  <- exp(log_transition_prob_old_new - log_transition_prob_new_old +  candidate_log_prob - old_log_prob)
  if(runif(1)<=accept_ratio){
    return(candidate_M)
  }
  else{
    return(M)
  }
} 


#' Update Theta Parameter in Joint MCMC Sampler
#'
#' Performs Metropolis-Hastings step to update the theta parameter in joint sampler.
#'
#' @param z1 First clustering vector
#' @param z2 Second clustering vector
#' @param M Current value of M parameter
#' @param theta Current value of theta parameter
#' @param gamma.d Binary indicator vector
#' @param params Model parameters including metropolis_theta_sd and beta prior parameters
#' @param rcpp Logical; use Rcpp implementation
#'
#' @return Updated value of theta parameter
#' @keywords internal
update_theta <- function(z1, z2, M, theta, gamma.d, params,rcpp){
  list2env(params,.GlobalEnv)
  candidate_theta   <- truncnorm::rtruncnorm(1, a=0, b=1, mean = theta, sd = metropolis_theta_sd)
  
  z2_R <- z2[(gamma.d==1)]
  
  candidate_log_prob <-  get_log_prob_partition(z2, M, candidate_theta,rcpp) + get_log_prob_partition(z1, M, candidate_theta,rcpp) - get_log_prob_partition(z2_R, M, candidate_theta,rcpp) + dbeta(candidate_theta, alpha.theta, beta.theta, log = TRUE)
  old_log_prob       <-  get_log_prob_partition(z2, M, theta,rcpp) + get_log_prob_partition(z1, M, theta,rcpp) - get_log_prob_partition(z2_R, M, theta,rcpp) + dbeta(theta, alpha.theta, beta.theta, log = TRUE)
  
  log_transition_prob_old_new <- log(dtruncnorm(theta, mean = candidate_theta, sd = metropolis_theta_sd))
  log_transition_prob_new_old <- log(dtruncnorm(candidate_theta, mean = theta, sd = metropolis_theta_sd))
  
  accept_ratio  <- exp(log_transition_prob_old_new - log_transition_prob_new_old + candidate_log_prob - old_log_prob)
  if(runif(1)<=accept_ratio){
    return(candidate_theta)
  }
  else{
    return(theta)
  }
} 


#' Joint MCMC Sampler for Two-Layer Voronoi Clustering
#'
#' Performs MCMC sampling for joint inference on two clustering solutions with shared structure.
#' Uses binary indicators (gamma) to determine where the two clusterings share the same partition.
#'
#' @param D1 First distance or data matrix
#' @param D2 Second distance or data matrix
#' @param params List of MCMC parameters including hyperparam_lik_1 and hyperparam_lik_2 with:
#'   \itemize{
#'     \item z.init: Initial cluster allocations
#'     \item M.init, theta.init, alpha.init: Initial parameter values
#'     \item metropolis_M_sd, metropolis_theta_sd: Proposal standard deviations
#'     \item lambda.M, alpha.theta, beta.theta: Prior hyperparameters
#'     \item a_alpha, b_alpha: Beta prior parameters for alpha
#'   }
#' @param N.sim Integer number of MCMC iterations
#' @param verbose Logical; print iteration details (default FALSE)
#' @param rcpp Logical; use Rcpp implementations (default TRUE)
#' @param M.fixed Optional fixed M value; if NULL, M is sampled
#' @param theta.fixed Optional fixed theta value; if NULL, theta is sampled
#' @param alpha.fixed Optional fixed alpha value; if NULL, alpha is sampled
#' @param save_file Optional file path for saving MCMC chain to disk
#'
#' @return List of MCMC samples, each containing:
#'   \itemize{
#'     \item M, theta, alpha.d: Sampled parameters
#'     \item gamma.d: Binary indicators for shared structure
#'     \item z1, z2: Cluster allocations
#'     \item K1, K2: Number of clusters
#'     \item log_lik: Log-likelihood
#'     \item time: Elapsed time
#'   }
#'   
#' @export
mcmc_sampler_joint <- function(D1, D2, params, N.sim, 
                               verbose=FALSE, rcpp = TRUE, 
                               M.fixed = NULL, theta.fixed = NULL, alpha.fixed = NULL,
                               save_file = NULL){

  
  if(is.null(params$hyperparam_lik_1$repulsion)) params$hyperparam_lik_1$repulsion = FALSE
  if(is.null(params$hyperparam_lik_2$repulsion)) params$hyperparam_lik_2$repulsion = FALSE
  
  
  list2env(params,.GlobalEnv)
  n <- nrow(D1)
  samples <- list()
  
  gamma.d <- rep(0,n)
  z1 <- params$hyperparam_lik_1$z.init
  z2 <- params$hyperparam_lik_2$z.init
  
  M       = ifelse(is.null(M.fixed), M.init, M.fixed)
  theta   = ifelse(is.null(theta.fixed), theta.init, theta.fixed)
  alpha.d = ifelse(is.null(alpha.fixed), alpha.init, alpha.fixed)
  
  if(!is.null(save_file)){
    largeList::saveList(object = list(M=M, theta=theta, alpha.d=alpha.d, gamma.d=gamma.d, z1=z1, z2=z2), file = save_file, append = FALSE)
  }
  
  start = proc.time()
  pb = utils::txtProgressBar(min = 0, max = N.sim, initial = 0, style = 0) 
  for(i in 2:N.sim){
    
    #update z1
    z1 <- update_z1(z1,z2,gamma.d,M,theta,D1,params, rcpp)
  
    #update gamma
    #print(z1)
    #print(z2)
    #print(gamma.d)
    gamma.d <- update_gammas(z1, z2, M, theta, gamma.d, alpha.d, rcpp)
    
    #updating z2 
    #z2=z1 #CHECK IF THIS IS CORRECT
    z2 <- update_z2(z1,z2,gamma.d,M,theta,D2,params, rcpp)
    
    #update alpha
    if(is.null(alpha.fixed)){
      alpha.d      <- rbeta(1, a_alpha+sum(gamma.d), b_alpha+n-sum(gamma.d)) 
    }

    #update M
    if(is.null(M.fixed)){
      M            <- update_M(z1, z2, M, theta, gamma.d, params, rcpp) 
    }

    #update theta
    if(is.null(theta.fixed)){
      theta        <- update_theta(z1, z2, M, theta, gamma.d, params, rcpp) 
    }

    #get log-likelihood for WAIC
    log_lik <- get_log_prob_D(D1, z1,  params$hyperparam_lik_1,rcpp) + get_log_prob_D(D2, z2,  params$hyperparam_lik_2,rcpp)


    samples[[length(samples)+1]] <- list(M=M, theta=theta,
                                         alpha.d=alpha.d, gamma.d=gamma.d,
                                         z1=z1, z2=z2, 
                                         K1 = length(unique(z1)), K2 = length(unique(z2)), 
                                         log_lik = log_lik,
                                         time = proc.time() - start)
    
    utils::setTxtProgressBar(pb,i)
    if(verbose){
      print(paste0("iteration ", i))
      print(z1)
      print(z2)
      print(alpha.d)
      print(M)
      print(theta)
    }
    
    if(!is.null(save_file)){
      qs2::qs_save(object = list(M=M, theta=theta, alpha.d=alpha.d, gamma.d=gamma.d, z1=z1, z2=z2, log_lik = log_lik), file = save_file)
      # largeList::saveList(object = list(M=M, theta=theta, alpha.d=alpha.d, gamma.d=gamma.d, z1=z1, z2=z2, log_lik = log_lik), file = save_file, append = TRUE)
    }
    
  }  
  close(pb)
  return(samples) 
}
