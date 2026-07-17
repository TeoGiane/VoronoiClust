#pragma once

// Standard library includes
#include <memory>
#include <string>

// Rcpp includes
#include <RcppArmadillo.h>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "rcpp_unpackers.h"


// Likelihoods factory
std::shared_ptr<AbstractLikelihood> build_likelihood(const Rcpp::List & params);

// Priors factory
std::shared_ptr<AbstractPrior> build_prior(const Rcpp::List & params);
std::shared_ptr<AbstractMixturePrior> build_mixture_prior(const Rcpp::List & params);