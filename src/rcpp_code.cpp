#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]


// CPP function for PY model


// Function to calculate the log of the Dirichlet Process mixture likelihood
arma::mat compute_Lkt(arma::mat&  D, arma::vec z, List params){
  

  double delta1 = as<double>(params["delta1"]);
  double zeta   = as<double>(params["zeta"]);
  double delta2 = as<double>(params["delta2"]);
  double gamma  = as<double>(params["gamma"]);
  double mu     = as<double>(params["mu"]);
  double beta   = as<double>(params["beta"]);
  double repulsion   = as<bool>(params["repulsion"]);
  
  arma::vec unique_z = unique(z);
  int K = unique_z.n_elem;
  arma::mat log_Lkt = arma::zeros<arma::mat>(K, K); // Matrix initialization with zeros
  
  
  for (int k = 0; k < K; ++k) {
    arma::uvec index_select = arma::find(z == unique_z(k));
    
    if (index_select.n_elem > 1){
      arma::mat D_k = D.submat(index_select, index_select);
      //arma::uvec indices = arma::find(arma::trimatu(arma::ones<arma::umat>(D_k.n_rows), 1));
      //std::cout << D_k  << std::endl;
      
      arma::vec distances = D_k(trimatu_ind( size(D_k) , 1));
      distances           = distances.elem(find_finite(distances));    //remove NAs
      
      log_Lkt(k, k)  = mu * log(beta) + (delta1 - 1) * sum(log(distances)) +
        (-mu - delta1 * distances.size()) * log(beta + sum(distances)) -
        distances.size() * lgamma(delta1) +
        lgamma(mu + distances.size() * delta1 ) - lgamma(mu);
    }
    
  }
  
  if (K > 1 && repulsion) {
    for (int k = 0; k < K - 1; ++k) {
      arma::uvec index_select1 = arma::find(z == unique_z(k));
      for (int t = k + 1; t < K; ++t) {
        arma::uvec index_select2 = arma::find(z == unique_z(t));
        
        arma::mat D_kt = D.submat(index_select1, index_select2);
        arma::vec distances = arma::vectorise(D_kt);
        distances           = distances.elem(find_finite(distances));   //remove NAs 
        
        log_Lkt(k, t) = zeta * log(gamma) + (delta2 - 1) * sum(log(distances)) +
          (-zeta - delta2 * distances.size()) * log(gamma + sum(distances)) -
          distances.size() * lgamma(delta2 ) +
          lgamma(zeta + distances.size() * delta2 ) - lgamma(zeta );
      }
    }
  }
  
  
  return log_Lkt;
  
}

// [[Rcpp::export]]
List get_log_prob_D_rcpp_list(arma::mat&  D, arma::vec z, List params) {
  
  arma::mat L_kt = compute_Lkt(D,z,params);
  
  double log_lik = accu(L_kt);
  return List::create(Named("log_lik") = log_lik, Named("log_Lkt") = L_kt);
}

// [[Rcpp::export]]
double get_log_prob_D_rcpp(arma::mat&  D, arma::vec z, List params) {
  arma::mat L_kt = compute_Lkt(D,z,params);
  double log_lik = accu(L_kt);
  return log_lik;
}


// Function to calculate the logarithm of the subtraction of two numbers
double log_subor_rcpp(double x, int m, double alpha = 1) {
  if (m == 0) return log(1);
  if (m == 1) return log(x);
  
  double log_prob = log(x);
  for (int i = 1; i < m; ++i) {
    log_prob += log(x + i * alpha);
  }
  return log_prob;
}

// [[Rcpp::export]]
double get_log_prob_partition_rcpp(arma::vec z, double M, double theta) {
  if (z.n_elem == 0) return 0;
  
  arma::vec unique_z = unique(z);
  int K = unique_z.n_elem;
  arma::vec nk(K);
  
  for (int i = 0; i < K; ++i) {
    nk(i) = arma::sum(z == unique_z(i));
  }
  
  int n = z.n_elem;
  
  double log_prob = log_subor_rcpp(M + theta, K - 1, theta) - log_subor_rcpp(M + 1, n - 1);
  
  for (int i = 0; i < K; ++i) {
    log_prob += log_subor_rcpp(1 - theta, nk(i) - 1);
  }
  
  return log_prob;
}

// Function to check compatibility between partitions c1 and c2
// bool check_same_partition_rcpp(arma::vec c1, arma::vec c2) {
//   int n1 = c1.size();
//   int n2 = c2.size();
//   
//   if (n1 != n2) return false;
//   if (n1 == 0) return true;
//   
//   arma::vec vec1 = arma::ones<arma::vec>(c1.size() + 2) * -1;
//   vec1.subvec(2, vec1.size() - 1) = c1;
//   
//   arma::vec vec2 = arma::ones<arma::vec>(c2.size() + 2) * -1;
//   vec2.subvec(2, vec2.size() - 1) = c2;
//   
//   // Obtain environment containing function
//   Rcpp::Environment base("package:igraph"); 
//   
//   // Make function callable from C++
//   Rcpp::Function compare_R = base["compare"];    
//   
//   // Call the function and receive its list output
//   return compare_R(vec1,vec2);
// }

// [[Rcpp::export]]
bool check_same_partition_rcpp(arma::vec c1, arma::vec c2) {
  // Check if lengths are equal
  if (c1.n_elem != c2.n_elem) return false;
  // Check if both are empty
  if (c1.n_elem == 0) return true;
  
  // Get unique values in c1
  arma::vec unique_values_c1 = arma::unique(c1);
  
  // Loop over unique values in c1
  for (size_t j = 0; j < unique_values_c1.n_elem; ++j) {
    double value = unique_values_c1(j);
    arma::uvec indices = arma::find(c1 == value);
    // If there is only one index, continue to the next unique value
    if (indices.n_elem == 1) continue;
    double value_check = c2(indices(0));
    // Check if subsequent indices have the same corresponding values in c2
    for (size_t i = 1; i < indices.n_elem; ++i) {
      if (c2(indices(i)) != value_check) return false;
    }
  }
  return true;
}

// Function to remove the i-th element from an arma::vec
arma::vec remove_element_rcpp(arma::vec z, int i) {
  // Check if i is within the valid range
  if (i <= 0 || i > z.n_elem) {
    Rcpp::stop("Index out of bounds");
  }
  
  // Create a subvector excluding the i-th element
  arma::vec z_without_i = arma::join_cols(z.head(i - 1), z.tail(z.n_elem - i));
  
  return z_without_i;
}

// [[Rcpp::export]]
arma::vec update_gammas_rcpp(arma::vec z1, arma::vec z2, double M, double theta, arma::vec gamma_d, double alpha_d) {
  
  int n = z1.n_elem;
  
  
  for (int i = 0; i < n; ++i) {
    double old_gammai = gamma_d(i);
    arma::vec z2_R_pi = z2.elem(find(gamma_d == 1 || arma::regspace(1, n) == (i +1)));
    arma::vec z2_R_mi = z2.elem(find(gamma_d == 1 && arma::regspace(1, n) != (i +1)));
    
    
    double ratio;
    if (z2_R_mi.n_elem == 0) {
      ratio = 1;
    } else {
      ratio = exp(get_log_prob_partition_rcpp(z2_R_pi, M, theta) - get_log_prob_partition_rcpp(z2_R_mi, M, theta));
    }
    
    double prob = alpha_d / (alpha_d + (1 - alpha_d) * ratio);
    
    //std::cout << prob << std::endl;
    
    double u = arma::randu();
    gamma_d(i) = u < prob ? 1 : 0; //possibily a mistake here, wath out!!
    
    
    if ((old_gammai == 0) && (gamma_d(i) == 1)) {
      bool compatibility = check_same_partition_rcpp(z1.elem(find(gamma_d == 1 || arma::regspace(1, n) == (i + 1))),
                                                     z2.elem(find(gamma_d == 1 || arma::regspace(1, n) == (i + 1))));
      
      if (!compatibility) {
        gamma_d(i) = 0;
        //std::cout << "not compatible - remove this comment" << std::endl;
      }
    }
  }
  
  return gamma_d;
}


// [[Rcpp::export]]
arma::vec  get_log_prob_allocations_comp_check_rcpp(arma::vec z1, arma::vec z2, arma::vec gamma_d, int i, double M, double theta, arma::mat D, Rcpp::List params) {
  
  arma::vec new_vec = remove_element_rcpp(z1, i);
  arma::vec unique_z1 = unique(new_vec);
  int K_mi = unique_z1.n_elem;
  arma::vec log_probs(K_mi + 1);
  
  for (int k = 0; k < log_probs.size(); ++k) {
    arma::vec z1_temp = z1;
    z1_temp(i - 1) = k + 1;
    
    bool compatibility = check_same_partition_rcpp(z1_temp.elem(find(gamma_d == 1)), z2.elem(find(gamma_d == 1)));
    
    if (!compatibility) {
      log_probs(k) = -arma::datum::inf;
      //  Rcpp::Rcout << "Compatibility check 2 -- delete this" << std::endl;
    } else {
      log_probs(k) = get_log_prob_partition_rcpp(z1_temp, M, theta) + get_log_prob_D_rcpp(D, z1_temp, params["hyperparam_lik_1"]);
    }
  }
  
  return log_probs;
}



// [[Rcpp::export]]
arma::vec get_log_prob_allocations_rcpp(arma::vec z, int i, double M, double theta, arma::mat D, List params) {
  
  arma::vec new_vec = remove_element_rcpp(z, i);
  arma::vec unique_z = unique(new_vec);
  int K_mi = unique_z.n_elem;
  
  arma::vec log_probs(K_mi + 1);
  
  for (int k = 0; k < log_probs.size(); ++k) {
    arma::vec z_temp = z;
    z_temp(i - 1) = k + 1; // Replace element i with k+1
    log_probs(k) = get_log_prob_partition_rcpp(z_temp, M, theta) + get_log_prob_D_rcpp(D, z_temp, params["hyperparam_lik_2"]);
  }
  
  return log_probs;
}

// [[Rcpp::export]]
arma::vec get_log_prob_allocations_marginal_rcpp(arma::vec z, int i, double M, double theta, arma::mat D, List params) {
  
  arma::vec new_vec = remove_element_rcpp(z, i);
  arma::vec unique_z = unique(new_vec);
  int K_mi = unique_z.n_elem;
  
  arma::vec log_probs(K_mi + 1);
  
  for (int k = 0; k < log_probs.size(); ++k) {
    arma::vec z_temp = z;
    z_temp(i - 1) = k + 1; // Replace element i with k+1
    log_probs(k) = get_log_prob_partition_rcpp(z_temp, M, theta) + get_log_prob_D_rcpp(D, z_temp, params);
  }
  
  return log_probs;
}


// [[Rcpp::export]]
double telescopic_dependence_rcpp(const arma::vec& z1, const arma::vec& z2) {
  int n = z1.size();
  double a1 = 0, a2 = 0, a3 = 0, a4 = 0, sum = 0;
  
  for(int i = 0; i < (n - 1); i++) {
    for(int j = i + 1; j < n; j++) {
      if(z2(i) == z2(j) && z1(i) == z1(j)) a1 += 1; 
      if(z1(i) == z1(j))                  a2 += 1; 
      if(z2(i) == z2(j) && z1(i) != z1(j)) a3 += 1; 
      if(z1(i) != z1(j))                  a4 += 1; 
      sum += 1;
    }
  }
  
  a1 /= sum;
  a2 /= sum;
  a3 /= sum;
  a4 /= sum;
  
  double P1 = a1 / a2;
  double P2 = a3 / a4;
  
  return ((P1 - P2) / P1);
}

// Function to update oracle_coclustering and oracle_posterior
// [[Rcpp::export]]

Rcpp::List updateOracle_rcpp(arma::mat& oracle_posterior, arma::mat& oracle_coclustering,
                             const arma::mat& pnts1, const arma::mat& clust_centres1, const arma::mat& Cov1,
                             const arma::vec& tempprobs) {
  int N = pnts1.n_rows;
  int K = clust_centres1.n_rows;
  
  
  // Calculate posterior
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      arma::mat squared_diff = arma::square(pnts1.row(i) - clust_centres1.row(j));
      double exponent = -0.5 * arma::accu(squared_diff) / std::pow(Cov1(0, 0), 2);
      oracle_posterior(j, i) = tempprobs(j) * std::exp(exponent);
    }
    oracle_posterior.col(i) = oracle_posterior.col(i) / arma::sum(oracle_posterior.col(i));
  }
  
  // Update oracle_coclustering
  oracle_coclustering = oracle_coclustering + arma::trans(oracle_posterior) * oracle_posterior;
  
  // Return a list containing updated matrices
  return Rcpp::List::create(Rcpp::Named("oracle_posterior") = oracle_posterior,
                            Rcpp::Named("oracle_coclustering") = oracle_coclustering);
}


// Function to compute distances
// [[Rcpp::export]]
List sampleDistances_rcpp(IntegerVector z1, List hyperparams_lik) {
  // Extract parameters
  double mu = as<double>(hyperparams_lik["mu"]);
  double beta = as<double>(hyperparams_lik["beta"]);
  double zeta = as<double>(hyperparams_lik["zeta"]);
  double gamma = as<double>(hyperparams_lik["gamma"]);
  double delta1 = as<double>(hyperparams_lik["delta1"]);
  double delta2 = as<double>(hyperparams_lik["delta2"]);
  
  // Get unique cluster labels
  IntegerVector cluster_labels = unique(z1);
  int n_clust = cluster_labels.size();
  
  // Generate lambda and theta matrices
  arma::vec lambda = arma::randg<arma::vec>(n_clust, arma::distr_param(mu, 1.0 / beta));
  arma::mat theta = arma::randg<arma::mat>(n_clust, n_clust, arma::distr_param(zeta, 1.0 / gamma));
  
  int n = z1.size();
  
  // Initialize distance matrix and vectors
  arma::mat D(n, n, arma::fill::zeros);
  arma::vec within_distances;
  arma::vec between_distances;
  
  
  // Compute distances
  for(int i = 0; i < (n - 1); i++) {
    for(int j = (i + 1); j < n; j++) {
      int cluster_i = z1[i] - 1; // Adjusting cluster labels to 0-based indexing
      int cluster_j = z1[j] - 1; // Adjusting cluster labels to 0-based indexing
      
      if (z1[i] == z1[j]) {
        D(i, j) = arma::randg(arma::distr_param(delta1, 1.0 / lambda(cluster_i)));
        D(j, i) = D(i, j);
        within_distances.resize(within_distances.n_elem + 1); // Resize vector
        within_distances(within_distances.n_elem - 1) = D(i, j); // Assign value
      } else {
        D(i, j) = arma::randg(arma::distr_param(delta2, 1.0 / theta(cluster_i, cluster_j)));
        D(j, i) = D(i, j);
        between_distances.resize(between_distances.n_elem + 1); // Resize vector
        between_distances(between_distances.n_elem - 1) = D(i, j); // Assign value
      }
    }
  }
  
  // Return as a list
  return List::create(_["D"] = D, _["within_distances"] = within_distances, _["between_distances"] = between_distances);
}

// Update z1
// [[Rcpp::export]]
arma::vec update_z1_rcpp(arma::vec z1, arma::vec z2, arma::vec gamma_d, double M, double theta, arma::mat D1, List params){
  
  Function update_z_rcpp("update_z");
  int n = z1.n_elem;
  
  int j;
  for (unsigned int i = 0; i < n; ++i) {
    j = i+1;                                  //return to R indexation
    arma::vec log_probs = get_log_prob_allocations_comp_check_rcpp(z1,z2,gamma_d,j,M,theta,D1,params);
    z1 = as<arma::vec>(update_z_rcpp(Named("z", z1), Named("j", j), Named("log_probs", log_probs)));
  }
  
  return z1;
}

// Update z2
// [[Rcpp::export]]
arma::vec update_z2_rcpp(arma::vec z1, arma::vec z2, arma::vec gamma_d, double M, double theta, arma::mat D2, List params){
  
  Function update_z_rcpp("update_z");
  
  // Find indices where gamma_d == 0
  arma::uvec update_indices = find(gamma_d == 0) + 1; // Adding 1 because R is 1-indexed
  
  int j;
  arma::vec log_probs;
  for (unsigned int i = 0; i < update_indices.n_elem; ++i) {
    
    j = update_indices(i);
    
    // Calculate log probabilities for allocations
    log_probs = get_log_prob_allocations_rcpp(z2, j, M, theta, D2, params);
    
    // Update z2 based on calculated log probabilities
    z2 = as<arma::vec>(update_z_rcpp(Named("z", z2), Named("j", j), Named("log_probs", log_probs)));
  }
  
  
  return z2;
}

// CPP function for tesselation clustering

// Assume that z is ordered 1 1 3 2 2 and not 1 1 5 7 7, and that 1 corresponds to the first medoid that shows up in gamma, 2 corresponds to the second medoid in gamma, etc...
// [[Rcpp::export]]
double get_log_prob_D_tesselation_2_rcpp(const arma::mat& D, const arma::uvec& z, const arma::uvec& centers, bool repulsion, const List& params) {
  // Extract parameters from the params list
  double mu = as<double>(params["mu"]);
  double beta = as<double>(params["beta"]);
  double delta1 = as<double>(params["delta1"]);
  double theta = as<double>(params["theta"]);
  //double zeta = as<double>(params["zeta"]);
  //double gamma = as<double>(params["gamma"]);
  double delta2 = as<double>(params["delta2"]);
  
  unsigned int K = centers.n_elem;
  
  double log_lik = 0;

  for (unsigned int k = 0; k < K; ++k) {
    arma::uvec index_select = find(z == (k + 1));  // Increment by 1 to match R's 1-based index
    if (index_select.n_elem == 1) continue;
    arma::rowvec selected_row = D.row(centers[k]-1);         // Get the row corresponding to the center
    arma::vec distances = selected_row.elem(index_select); // Extract the elements by index_select
    distances = distances.elem(find(distances > 0));
    unsigned int n_d = distances.n_elem;
    
    log_lik +=  mu * log(beta) + (delta1 - 1) * sum(log(distances)) +
      (-mu - delta1 * n_d) * log(beta + sum(distances)) -
      n_d * lgamma(delta1) +
      lgamma(mu + n_d * delta1 ) - lgamma(mu); //optimize this some terms can be out of sum....
    
  }
  
  //unsigned int n         = z.n_elem;
  //double cal_const =  (n - K)/(K*(K-1)/2);
  //double log_norm_constant = -delta2*cal_const*log(theta) - (-1 + cal_const - delta2*cal_const)*(log(cal_const) + log(theta)) +  cal_const*lgamma(delta2) - lgamma(1 + (-1 + delta2)*cal_const);
    
  if (repulsion && K > 1) {
    arma::mat D_centers = D.submat(centers-1, centers-1);
    
    arma::vec distances = D_centers.elem(find(trimatu(D_centers, 1)));
    
    unsigned int n_d = distances.n_elem;
    
    log_lik +=   n_d*delta2*log(theta) - n_d*lgamma(delta2) -theta*sum(distances) + (delta2-1)*sum(log(distances));
    
    // old gamma-gamma likelihood
    //zeta * log(gamma) + (delta2 - 1) * sum(log(distances)) - (zeta + delta2 * distances.n_elem) * log(gamma + sum(distances))
    //    - distances.n_elem * lgamma(delta2) + lgamma(zeta + distances.n_elem * delta2) - lgamma(zeta);
    
    //calibrated likelihood
    //log_lik +=   cal_const*(n_d*delta2*log(theta) - n_d*lgamma(delta2) -theta*sum(distances) + (delta2-1)*sum(log(distances))) + log_norm_constant;
    
  }
  
  return log_lik;
}

// [[Rcpp::export]]
double get_log_prob_D_tesselation_rcpp(const arma::mat& D, const arma::vec& z, const arma::uvec& centers, bool repulsion, const List& params) {
  
  
  double delta1 = as<double>(params["delta1"]);
  double zeta   = as<double>(params["zeta"]);
  double delta2 = as<double>(params["delta2"]);
  double gamma  = as<double>(params["gamma"]);
  double mu     = as<double>(params["mu"]);
  double beta   = as<double>(params["beta"]);
  
  
  arma::vec unique_z = unique(z);
  int K = unique_z.n_elem;
  arma::mat log_Lkt = arma::zeros<arma::mat>(K, K); // Matrix initialization with zeros
  
  
  for (int k = 0; k < K; ++k) {
    arma::uvec index_select = arma::find(z == unique_z(k));
    
    if (index_select.n_elem > 1){
      arma::mat D_k = D.submat(index_select, index_select);
      //arma::uvec indices = arma::find(arma::trimatu(arma::ones<arma::umat>(D_k.n_rows), 1));
      //std::cout << D_k  << std::endl;
      
      arma::vec distances = D_k(trimatu_ind( size(D_k) , 1));
      distances           = distances.elem(find_finite(distances));    //remove NAs
      
      log_Lkt(k, k)  = mu * log(beta) + (delta1 - 1) * sum(log(distances)) +
        (-mu - delta1 * distances.size()) * log(beta + sum(distances)) -
        distances.size() * lgamma(delta1) +
        lgamma(mu + distances.size() * delta1 ) - lgamma(mu);
    }
    
  }
  
if (K > 1 && repulsion) {
  for (int k = 0; k < K - 1; ++k) {
    arma::uvec index_select1 = arma::find(z == unique_z(k));
   for (int t = k + 1; t < K; ++t) {
      arma::uvec index_select2 = arma::find(z == unique_z(t));

      arma::mat D_kt = D.submat(index_select1, index_select2);
       arma::vec distances = arma::vectorise(D_kt);
      distances           = distances.elem(find_finite(distances));   //remove NAs

      log_Lkt(k, t) = zeta * log(gamma) + (delta2 - 1) * sum(log(distances)) +
         (-zeta - delta2 * distances.size()) * log(gamma + sum(distances)) -
        distances.size() * lgamma(delta2 ) +
         lgamma(zeta + distances.size() * delta2 ) - lgamma(zeta );
     }
   }
 }
  
  
  return accu(log_Lkt);
}



