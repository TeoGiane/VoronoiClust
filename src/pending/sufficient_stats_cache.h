#pragma once

// Standard library includes
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <utility>

// Rcpp includes
#include <RcppArmadillo.h>

// Local includes
#include "likelihoods.h"

/*
 * ============================================================================
 * Incremental sufficient-statistics caches for the tessellation likelihoods.
 * ============================================================================
 *
 * Both QuadraticTessellationLikelihood and LinearTessellationLikelihood are
 * sums of per-cluster (or per-cluster-pair) terms that are themselves closed
 * -form functions of a handful of running sums over pairwise distances:
 * (n_pairs, sum_d, sum_log_d). When a single point changes cluster label,
 * only a bounded set of these running sums change -- so the *log-lpdf delta*
 * of a single-point move can be computed and applied in O(n) (one pass over
 * the distance matrix, to find the moved point's distances to every other
 * currently-labelled point) instead of recomputing the O(n^2) likelihood
 * from scratch.
 *
 * KEY FACT (see design discussion): applying this single-point update
 * sequentially, one point at a time, for a batch of several points that all
 * move "simultaneously" (as happens in a Voronoi birth/death/move proposal)
 * produces the EXACT same total as a full recompute -- including newly
 * formed within-cluster pairs among two points that both move into the same
 * destination cluster together. This holds regardless of processing order,
 * because each step buckets by the *current* (already partially updated)
 * labels. So there is no need for a separate "batch" algorithm: a loop over
 * point_to_clusters() + commit_move() per mover is exact.
 *
 * Two cache classes are provided:
 *   - QuadraticStatsCache / QuadraticStatsTrialCache, for the O(n_c^2)
 *     within + O(n_c * n_t) between pairwise likelihood.
 *   - LinearStatsCache, for the medoid/centre-anchored likelihood, valid
 *     ONLY when centres are externally fixed (i.e. VoronoiSampler, where
 *     `centers` is always supplied) -- NOT for PYSampler's implicit
 *     per-cluster medoid search, where the "centre" identity can itself
 *     change as points move, breaking the O(1)-per-point update.
 * ============================================================================
 */

struct PairStat {
    long n_pairs = 0;
    double sum_d = 0.0;
    double sum_log_d = 0.0;
};

// Per-point aggregate distance stats to every currently active cluster,
// bucketed by the CURRENT label of the other endpoint. Cluster id -> stat.
using ClusterAggMap = std::unordered_map<int, PairStat>;

inline uint64_t pack_pair_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}
inline std::pair<int,int> unpack_pair_key(uint64_t key) {
    return { static_cast<int>(key >> 32), static_cast<int>(key & 0xFFFFFFFFu) };
}

/* --------------------------- Quadratic cache --------------------------- */

class QuadraticStatsCache {
  protected:
    QuadraticLikelihoodParams params;
    double const_within, const_between;
    double cached_lpdf = 0.0;
    std::unordered_map<int, PairStat> within_stats;
    std::unordered_map<uint64_t, PairStat> between_stats;

    double f_within(const PairStat& s) const;
    double f_between(const PairStat& s) const;

  public:
    // Overridable storage accessors: QuadraticStatsTrialCache overrides
    // these to provide a copy-on-write overlay on top of a base cache,
    // WITHOUT ever mutating the base. All public methods below are written
    // in terms of these, never touching within_stats/between_stats directly,
    // so a single implementation of eval_move_delta/commit_move serves both
    // the persistent cache and any trial overlay built on top of it.
    // PUBLIC (not protected): QuadraticStatsTrialCache::fold_into() writes
    // into a `QuadraticStatsCache&` (the base type, not necessarily another
    // QuadraticStatsTrialCache), and C++'s protected-access rule only
    // allows a derived class to touch a base's protected members through an
    // object of its OWN (derived) type -- not through a plain base
    // reference. Keeping these public sidesteps that restriction.
    virtual PairStat get_within(int k) const;
    virtual PairStat get_between(int a, int b) const;
    virtual void put_within(int k, const PairStat& s);
    virtual void put_between(int a, int b, const PairStat& s);

  public:
    explicit QuadraticStatsCache(const QuadraticLikelihoodParams& p);
    virtual ~QuadraticStatsCache() = default;

    // O(n^2) full rebuild from the raw distance matrix. Call once at init,
    // and periodically thereafter (e.g. every few hundred iterations) to
    // correct floating-point drift accumulated by repeated incremental
    // add/subtract operations.
    void rebuild(const arma::mat& dist_matrix, const arma::uvec& allocs);

    double current_lpdf() const { return cached_lpdf; }
    void adjust_lpdf(double delta) { cached_lpdf += delta; }
    const QuadraticLikelihoodParams& get_params() const { return params; }

    // O(n): point m's aggregate distance stats to every cluster currently
    // present in `allocs` (bucketed by CURRENT label, excludes m itself).
    // Distances <= 0 are skipped, matching eval_lpdf's own convention.
    ClusterAggMap point_to_clusters(const arma::mat& dist_matrix, const arma::uvec& allocs, arma::uword m) const;

    // Read-only: log-lpdf delta of moving the point whose aggregate is `agg`
    // from cluster `from` to cluster `to`. Does not mutate the cache.
    double eval_move_delta(const ClusterAggMap& agg, int from, int to) const;

    // Mutates the cache (within/between stats + cached_lpdf) to reflect the
    // move. Call with the SAME `agg` that was passed to eval_move_delta (or
    // recomputed against the same pre-move `allocs` state).
    void commit_move(const ClusterAggMap& agg, int from, int to);

    // Rewrites every cluster id key (within_stats and between_stats) through
    // `id_map`. Needed after any relabeling that is NOT the identity map on
    // the cache's existing ids -- concretely, after a merge's
    // standardize_allocs() call, which closes the gap left by the removed
    // cluster and can shift every id above it down by one. A cluster id with
    // no entry in `id_map` is dropped (used for the now-empty merged-away
    // label, which id_map deliberately omits). O(K), not O(n).
    void remap_ids(const std::unordered_map<int,int>& id_map);
};

// Disposable copy-on-write overlay for scoring a batch of moves (a
// split-merge launch-state sweep, or a Voronoi birth/death/move candidate)
// without touching the persistent cache. Reads fall back to `base` for any
// entry not yet written locally; writes always land in the local overlay.
// `cached_lpdf` on a trial therefore tracks the DELTA relative to base, not
// an absolute lpdf, since it starts at 0 and every commit_move only ever
// adds (new - old) differences.
class QuadraticStatsTrialCache : public QuadraticStatsCache {
  private:
    const QuadraticStatsCache& base;

  public:
    PairStat get_within(int k) const override;
    PairStat get_between(int a, int b) const override;
    // put_within / put_between: inherited (write to this object's own maps).

    explicit QuadraticStatsTrialCache(const QuadraticStatsCache& base_cache);

    // Total lpdf delta accumulated by this trial relative to `base`.
    double delta_lpdf() const { return cached_lpdf; }

    // Folds this trial's touched entries + accumulated delta back into a
    // persistent cache. O(number of touched entries), call only on
    // acceptance of the batch of moves this trial scored.
    void fold_into(QuadraticStatsCache& target) const;
};

/* ----------------------------- Linear cache ----------------------------- */

// Valid ONLY when cluster centres are externally fixed and passed in (i.e.
// VoronoiSampler with `centers` always supplied). Within-cluster stats are
// per-cluster running sums of (centre -> member) spoke distances -- O(1) to
// update per point move, since no medoid search is involved. Between-cluster
// stats are a plain K x K matrix over the (small) set of active centres,
// trivially patched whenever the centre set changes.
class LinearStatsCache {
  private:
    LinearLikelihoodParams params;
    double const_within;
    double cached_lpdf = 0.0;
    // within_stats keyed by cluster id (== index into current centre list
    // is NOT assumed; keyed by the cluster label used in cluster_allocs).
    std::unordered_map<int, PairStat> within_stats;
    // between term depends only on the set of active centre POINT INDICES
    // (not cluster labels), since it is a function of centre-to-centre
    // distances. Cached as the total between contribution directly, since
    // recomputing it from the (small) centre list is already O(K^2).
    double between_lpdf = 0.0;

    double f_within(const PairStat& s) const;

  public:
    explicit LinearStatsCache(const LinearLikelihoodParams& p);

    // Full rebuild: O(n) for the within term (one spoke distance per point
    // to its cluster's fixed centre) + O(K^2) for the between term.
    void rebuild(const arma::mat& dist_matrix, const arma::uvec& allocs,
                 const std::vector<arma::uword>& centres);

    double current_lpdf() const { return cached_lpdf; }

    // O(1): add/remove a single point's spoke distance to/from a cluster's
    // running within-stat, given that cluster's fixed centre point index.
    void add_point(const arma::mat& dist_matrix, arma::uword point_idx, int cluster_id, arma::uword centre_idx);
    void remove_point(const arma::mat& dist_matrix, arma::uword point_idx, int cluster_id, arma::uword centre_idx);

    // O(K): recompute the between term from scratch given the current
    // centre list. Call whenever the centre set changes (birth/death/move).
    void recompute_between(const arma::mat& dist_matrix, const std::vector<arma::uword>& centres);

    // Snapshot / restore for scoring candidate proposals without commitment
    // (the between term and per-cluster within stats are both small, so a
    // full copy is cheap -- no need for the copy-on-write machinery used by
    // the quadratic cache).
    struct Snapshot {
        std::unordered_map<int, PairStat> within_stats;
        double between_lpdf;
        double cached_lpdf;
    };
    Snapshot snapshot() const { return { within_stats, between_lpdf, cached_lpdf }; }
    void restore(const Snapshot& s) { within_stats = s.within_stats; between_lpdf = s.between_lpdf; cached_lpdf = s.cached_lpdf; }
};
