#' @name prior_params
#' @title Prior Parameters
#' @description A list specifying the parameters for the prior distribution over the clustering structure.
#'
#' @section Tessellation Models (`mcmc_tessellation`):
#' \strong{SISTEMARE/CONTROLLARE}The prior(s) defined in this section can be used in \code{\link{mcmc_tessellation}} or as a building block for
#' \code{\link{mcmc_tessellation_multiview}}. The prior we define is a truncated geometric distribution on the number of
#' clusters. The PMF is given by:
#' \deqn{\pi(\boldsymbol{\gamma}) = \binom{N}{K}^{-1} \times \text{TGeom}(K;p) \ \ \ \ 0<K = |\boldsymbol{\gamma}|\leq N}
#' \describe{
#'   \item{\strong{truncated_geometric}}{Fixed parameters for the truncated geomteric prior. The parameter list has to contain the following items:
#'     \itemize{
#'       \item `type`: A string, must be `"truncated_geometric"`.
#'       \item `prob`: The success probability of the geometric distribution.
#'       \item `min`: The minimum number of clusters.
#'       \item `max`: The maximum number of clusters.
#'     }
#'   }
#' }
#'
#' @section Mixture Models:
#' The prior(s) defined in this section can be used in \code{\link{mcmc_PY}} or as a building block for
#' \code{\link{mcmc_PY_multiview}}. The EPPF of the Pitman Yor model is given by:
#' \deqn{\pi(n_1, \ldots, n_K) = \frac{[M+\theta]_{K-1 ; \theta}}{[M+1]_{N-1;1}} \prod_{i=1}^K[1-\varphi]_{n_i-1;1},}
#' where \eqn{M} is the concentration parameter and \eqn{\varphi} is the discount parameter. For the Pitman-Yor mixture models, 
#' the prior can be fixed or hierarchical.
#' \describe{
#'   \item{\strong{PY-fixed}}{Fixed paramers for the PY prior. The parameter list has to contain the following items:
#'     \itemize{
#'       \item \code{type}: Must be "PY-fixed".
#'       \item \code{discount}: The discount parameter \eqn{\varphi}.
#'       \item \code{concentration}: The concentration parameter \eqn{M}, where \eqn{M > -\varphi}.
#'     }
#'   }
#'   \item{\strong{PY-hierarchical}}{ 
#'     \itemize{
#'       \item \code{type}: Must be "PY-hierarchical".
#'       \item \code{discount_alpha}: Shape parameter alpha for the Beta prior on the discount parameter.
#'       \item \code{discount_beta}: Shape parameter beta for the Beta prior on the discount parameter.
#'       \item \code{concentration_shape}: Shape parameter for the Gamma prior on the concentration parameter.
#'       \item \code{concentration_rate}: Rate parameter for the Gamma prior on the concentration parameter.
#'     }
#'   }
#' }
NULL