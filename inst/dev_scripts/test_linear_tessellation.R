# library(VoronoiClust)
library(HPCVoronoiClust)
library(ggplot2)

# generate synthetic data
N <- 100
result <- VoronoiClust::generate_mixture(N = N, K = 10, M = 10, dim = 10, alpha = 1, compute.oracle = FALSE, sigma1 = 0.1, sigma2 = 0.1)
D = result$D1

# Specify Likelihood parameters via Empirical Bayes
lik_params = compute_EB_params(D, 10, result$clusts1, linear = TRUE, repulsion = TRUE)
# params$medoids.init <- round(seq(1,N,,4))
# params$z.init <- 1:N

# prior predictive check
# prior_data     <- marginal_prior_predictive_sampler(100, 300, params, M.fixed = 1, theta.fixed = 0.1)
# data.plot.same <- as.data.frame(rbind( cbind(samples = result$D1[outer(result$clusts1,result$clusts1, function(x,y) x == y) & upper.tri(diag(length(result$clusts1)))], type = "data" , coins = "Same die"),
#                                   cbind(samples = to_matrix(prior_data,"within_distances_1"), type = "prior predictive", coins = "Same die") ) )
# data.plot.different <- as.data.frame(rbind( cbind(samples = result$D1[outer(result$clusts1,result$clusts1, function(x,y) x != y) & upper.tri(diag(length(result$clusts1)))]  , type = "data", coins = "Different die" ),
#                                   cbind(samples = to_matrix(prior_data,"between_distances_1"), type = "prior predictive", coins = "Different die") ) )
# data.plot = rbind(data.plot.same, data.plot.different)

# plot data vs prior predictive
# data.plot$samples <- as.numeric(data.plot$samples)
# ggplot(data= data.plot) + #data=data.plot, aes(x=samples, fill=coins)) +
#   geom_density(aes(x=samples, fill=coins), alpha = 0.5) +
#   xlim(0,2) + facet_grid(type ~ .)

# fit model
# start <- proc.time()
# fit <- VoronoiClust::mcmc_tesselation(D = D, params = params, N.sim = 10000, verbose = FALSE)
# stop <- proc.time()
# print(stop - start)

# # check coclustering
# dat <- to_matrix(fit,"z")
# image(coclustering(dat))

# # check number of clusters traceplot
# dat <- to_matrix(fit,"K")
# plot.ts(dat)




# TEST NEW

# lik_params <- list(
#   "type" = "quadratic",
#   "shape_within" = params$delta1,
#   "prior_shape_within" = params$mu,
#   "prior_rate_within" = params$beta,
#   "shape_between" = params$delta2,
#   "prior_shape_between" = params$zeta,
#   "prior_rate_between" = params$gamma,
#   "repulsion" = TRUE
# )

# Specify prior parameters
prior_params <- list(
  "type" = "truncated_geometric",
  "min" = 0,
  "max" = nrow(D),
  "prob" = 0.25
)

# Specify algorithm parameters
algo_params <- list(
  "iterations" = 10000,
  "burnin" = 1000,
  "thinning" = 1
)

# Run MCMC
start <- proc.time()
tmp <- mcmc_tessellation(D, lik_params, prior_params, algo_params)
stop <- proc.time()
print(stop-start)

# Visualization (siple check)
image(D)
image(salso::psm(tmp$cluster_allocs))

# # Test new PY sampler
# lik_params <- list(
#   "type" = "quadratic",
#   "shape_within" = params$delta1,
#   "prior_shape_within" = params$mu,
#   "prior_rate_within" = params$beta,
#   "shape_between" = params$delta2,
#   "prior_shape_between" = params$zeta,
#   "prior_rate_between" = params$gamma,
#   "repulsion" = TRUE
# )

# prior_params <- list(
#   "type" = "PY-fixed",
#   "concentration" = 1.0,
#   "discount" = 0.1
# )

# prior_params <- list(
#   "type" = "PY-hierarchical",
#   "discount_alpha" = 1.0,
#   "discount_beta" = 1.0,
#   "concentration_shape" = 2.0,
#   "concentration_rate" = 2.0
# )

# algo_params <- list(
#   "iterations" = 10000,
#   "burnin" = 1000,
#   "thinning" = 1
# )

# start <- proc.time()
# tmp <- HPCVoronoiClust::mcmc_PY(D, lik_params, prior_params, algo_params)
# stop <- proc.time()
# print(stop-start)
