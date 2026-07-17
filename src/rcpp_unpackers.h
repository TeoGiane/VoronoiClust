#pragma once

// Rcpp includes
#include <RcppArmadillo.h>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "sampler_types.h"
#include "py_sampler.h"


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
    inline PYFixedParams as(SEXP x) {
        Rcpp::List lst(x);
        return PYFixedParams {
            Rcpp::as<double>(lst["discount"]),
            Rcpp::as<double>(lst["concentration"])
        };
    };

    template <>
    inline PYHierarchicalParams as(SEXP x) {
        Rcpp::List lst(x);
        return PYHierarchicalParams {
            Rcpp::as<double>(lst["discount_alpha"]),
            Rcpp::as<double>(lst["discount_beta"]),
            Rcpp::as<double>(lst["concentration_shape"]),
            Rcpp::as<double>(lst["concentration_rate"])
        };
    };

    template <>
    inline AlgorithmParams as(SEXP x) {
        Rcpp::List lst(x);
        return AlgorithmParams {
            Rcpp::as<size_t>(lst["iterations"]),
            Rcpp::as<size_t>(lst["burnin"]),
            Rcpp::as<size_t>(lst["thinning"]),
            extract_default<int>(lst, "init_n_clust", 5),
            extract_default<double>(lst, "tempering", 0.0),
            extract_default<int>(lst, "random_seed", 20260714),
            extract_default<bool>(lst, "debug", false)
        };
    };

    template<>
    inline MixtureAlgorithmParams as(SEXP x) {
        Rcpp::List lst(x);
        return MixtureAlgorithmParams {
            Rcpp::as<size_t>(lst["iterations"]),
            Rcpp::as<size_t>(lst["burnin"]),
            Rcpp::as<size_t>(lst["thinning"]),
            extract_default<int>(lst, "init_n_clust", 5),
            extract_default<int>(lst, "n_sweeps", 1),
            extract_default<double>(lst, "target_acc_rate", 0.44),
            extract_default<double>(lst, "adapt_decay", 0.6),
            extract_default<int>(lst, "random_seed", 20260714),
            extract_default<bool>(lst, "debug", false)
        };
    };
} // namespace Rcpp