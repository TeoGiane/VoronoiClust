#pragma once

// Rcpp includes
#include <RcppArmadillo.h>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "voronoi_sampler_types.h"


namespace Rcpp {

    template <typename T>
    inline T extract_default(const Rcpp::List & lst, const std::string & name, T default_value) {
        if (lst.containsElementNamed(name.c_str())) {
            SEXP elem = lst[name];
            if (!Rf_isNull(elem)) {
                return Rcpp::as<T>(elem);
            }
        }
        return default_value;
    };

    template <>
    inline QuadraticLikelihoodParams as(SEXP x) {
        Rcpp::List lst(x);
        return QuadraticLikelihoodParams {
            Rcpp::as<double>(lst["shape_within"]),
            Rcpp::as<double>(lst["prior_shape_within"]),
            Rcpp::as<double>(lst["prior_rate_within"]),
            Rcpp::as<double>(lst["shape_between"]),
            Rcpp::as<double>(lst["prior_shape_between"]),
            Rcpp::as<double>(lst["prior_rate_between"]),
            Rcpp::as<bool>(lst["repulsion"])
        };
    };

    template <>
    inline TruncatedGeometricParams as(SEXP x) {
        Rcpp::List lst(x);
        return TruncatedGeometricParams {
            Rcpp::as<unsigned int>(lst["min"]),
            Rcpp::as<unsigned int>(lst["max"]),
            Rcpp::as<double>(lst["prob"])
        };
    };

    template <>
    inline AlgorithmParams as(SEXP x) {
        Rcpp::List lst(x);
        return AlgorithmParams {
            extract_default<int>(lst, "iterations", 10000),
            extract_default<int>(lst, "burnin", 1000),
            extract_default<int>(lst, "thinning", 1),
            extract_default<double>(lst, "tempering", 0.0),
            extract_default<int>(lst, "init_n_clust", 5),
            extract_default<int>(lst, "random_seed", 20260714),
            extract_default<bool>(lst, "debug", false)
        };
    };

} // namespace Rcpp