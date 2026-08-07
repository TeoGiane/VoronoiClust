#include "sufficient_stats_cache.h"

/* ============================== Quadratic ============================== */

QuadraticStatsCache::QuadraticStatsCache(const QuadraticLikelihoodParams& p) : params(p) {
    const_within  = params.prior_shape_within  * std::log(params.prior_rate_within)  - std::lgamma(params.prior_shape_within);
    const_between = params.prior_shape_between * std::log(params.prior_rate_between) - std::lgamma(params.prior_shape_between);
}

double QuadraticStatsCache::f_within(const PairStat& s) const {
    if (s.n_pairs <= 0) return 0.0;
    return const_within
         + (params.shape_within - 1.0) * s.sum_log_d
         + (-params.prior_shape_within - params.shape_within * s.n_pairs) * std::log(params.prior_rate_within + s.sum_d)
         - s.n_pairs * std::lgamma(params.shape_within)
         + std::lgamma(params.prior_shape_within + s.n_pairs * params.shape_within);
}

double QuadraticStatsCache::f_between(const PairStat& s) const {
    if (!params.repulsion || s.n_pairs <= 0) return 0.0;
    return const_between
         + (params.shape_between - 1.0) * s.sum_log_d
         + (-params.prior_shape_between - params.shape_between * s.n_pairs) * std::log(params.prior_rate_between + s.sum_d)
         - s.n_pairs * std::lgamma(params.shape_between)
         + std::lgamma(params.prior_shape_between + s.n_pairs * params.shape_between);
}

PairStat QuadraticStatsCache::get_within(int k) const {
    auto it = within_stats.find(k);
    return (it != within_stats.end()) ? it->second : PairStat{};
}
PairStat QuadraticStatsCache::get_between(int a, int b) const {
    auto it = between_stats.find(pack_pair_key(a, b));
    return (it != between_stats.end()) ? it->second : PairStat{};
}
void QuadraticStatsCache::put_within(int k, const PairStat& s) { within_stats[k] = s; }
void QuadraticStatsCache::put_between(int a, int b, const PairStat& s) { between_stats[pack_pair_key(a, b)] = s; }

void QuadraticStatsCache::rebuild(const arma::mat& dist_matrix, const arma::uvec& allocs) {
    within_stats.clear();
    between_stats.clear();
    size_t n = allocs.n_elem;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double d = dist_matrix(i, j);
            if (d <= 0.0) continue;
            int ci = allocs(i), cj = allocs(j);
            double log_d = std::log(d);
            if (ci == cj) {
                PairStat& s = within_stats[ci];
                s.n_pairs++; s.sum_d += d; s.sum_log_d += log_d;
            } else if (params.repulsion) {
                PairStat& s = between_stats[pack_pair_key(ci, cj)];
                s.n_pairs++; s.sum_d += d; s.sum_log_d += log_d;
            }
        }
    }
    cached_lpdf = 0.0;
    for (const auto& kv : within_stats)  cached_lpdf += f_within(kv.second);
    for (const auto& kv : between_stats) cached_lpdf += f_between(kv.second);
}

ClusterAggMap QuadraticStatsCache::point_to_clusters(const arma::mat& dist_matrix, const arma::uvec& allocs, arma::uword m) const {
    ClusterAggMap agg;
    for (arma::uword x = 0; x < allocs.n_elem; ++x) {
        if (x == m) continue;
        double d = dist_matrix(m, x);
        if (d <= 0.0) continue;
        PairStat& s = agg[allocs(x)];
        s.n_pairs++; s.sum_d += d; s.sum_log_d += std::log(d);
    }
    return agg;
}

double QuadraticStatsCache::eval_move_delta(const ClusterAggMap& agg, int from, int to) const {
    if (from == to) return 0.0;
    auto it_f = agg.find(from);
    auto it_t = agg.find(to);
    PairStat d_mf = (it_f != agg.end()) ? it_f->second : PairStat{};
    PairStat d_mt = (it_t != agg.end()) ? it_t->second : PairStat{};

    PairStat old_wf = get_within(from);
    PairStat old_wt = get_within(to);
    PairStat new_wf{old_wf.n_pairs - d_mf.n_pairs, old_wf.sum_d - d_mf.sum_d, old_wf.sum_log_d - d_mf.sum_log_d};
    PairStat new_wt{old_wt.n_pairs + d_mt.n_pairs, old_wt.sum_d + d_mt.sum_d, old_wt.sum_log_d + d_mt.sum_log_d};
    double delta = (f_within(new_wf) - f_within(old_wf)) + (f_within(new_wt) - f_within(old_wt));

    if (params.repulsion) {
        PairStat old_bft = get_between(from, to);
        PairStat new_bft{
            old_bft.n_pairs   + d_mf.n_pairs   - d_mt.n_pairs,
            old_bft.sum_d     + d_mf.sum_d     - d_mt.sum_d,
            old_bft.sum_log_d + d_mf.sum_log_d - d_mt.sum_log_d
        };
        delta += f_between(new_bft) - f_between(old_bft);

        for (const auto& kv : agg) {
            int C = kv.first;
            if (C == from || C == to) continue;
            const PairStat& d_mc = kv.second;
            PairStat old_bfc = get_between(from, C);
            PairStat old_btc = get_between(to, C);
            PairStat new_bfc{old_bfc.n_pairs - d_mc.n_pairs, old_bfc.sum_d - d_mc.sum_d, old_bfc.sum_log_d - d_mc.sum_log_d};
            PairStat new_btc{old_btc.n_pairs + d_mc.n_pairs, old_btc.sum_d + d_mc.sum_d, old_btc.sum_log_d + d_mc.sum_log_d};
            delta += (f_between(new_bfc) - f_between(old_bfc)) + (f_between(new_btc) - f_between(old_btc));
        }
    }
    return delta;
}

void QuadraticStatsCache::commit_move(const ClusterAggMap& agg, int from, int to) {
    if (from == to) return;
    auto it_f = agg.find(from);
    auto it_t = agg.find(to);
    PairStat d_mf = (it_f != agg.end()) ? it_f->second : PairStat{};
    PairStat d_mt = (it_t != agg.end()) ? it_t->second : PairStat{};

    PairStat wf = get_within(from);
    cached_lpdf -= f_within(wf);
    wf.n_pairs -= d_mf.n_pairs; wf.sum_d -= d_mf.sum_d; wf.sum_log_d -= d_mf.sum_log_d;
    cached_lpdf += f_within(wf);
    put_within(from, wf);

    PairStat wt = get_within(to);
    cached_lpdf -= f_within(wt);
    wt.n_pairs += d_mt.n_pairs; wt.sum_d += d_mt.sum_d; wt.sum_log_d += d_mt.sum_log_d;
    cached_lpdf += f_within(wt);
    put_within(to, wt);

    if (params.repulsion) {
        PairStat bft = get_between(from, to);
        cached_lpdf -= f_between(bft);
        bft.n_pairs   += d_mf.n_pairs   - d_mt.n_pairs;
        bft.sum_d     += d_mf.sum_d     - d_mt.sum_d;
        bft.sum_log_d += d_mf.sum_log_d - d_mt.sum_log_d;
        cached_lpdf += f_between(bft);
        put_between(from, to, bft);

        for (const auto& kv : agg) {
            int C = kv.first;
            if (C == from || C == to) continue;
            const PairStat& d_mc = kv.second;

            PairStat bfc = get_between(from, C);
            cached_lpdf -= f_between(bfc);
            bfc.n_pairs -= d_mc.n_pairs; bfc.sum_d -= d_mc.sum_d; bfc.sum_log_d -= d_mc.sum_log_d;
            cached_lpdf += f_between(bfc);
            put_between(from, C, bfc);

            PairStat btc = get_between(to, C);
            cached_lpdf -= f_between(btc);
            btc.n_pairs += d_mc.n_pairs; btc.sum_d += d_mc.sum_d; btc.sum_log_d += d_mc.sum_log_d;
            cached_lpdf += f_between(btc);
            put_between(to, C, btc);
        }
    }
}

void QuadraticStatsCache::remap_ids(const std::unordered_map<int,int>& id_map) {
    std::unordered_map<int, PairStat> new_within;
    for (const auto& kv : within_stats) {
        auto it = id_map.find(kv.first);
        if (it != id_map.end()) new_within[it->second] = kv.second;
        // else: id had no entry (e.g. the merged-away label) -- drop it.
    }
    std::unordered_map<uint64_t, PairStat> new_between;
    for (const auto& kv : between_stats) {
        auto [a, b] = unpack_pair_key(kv.first);
        auto it_a = id_map.find(a);
        auto it_b = id_map.find(b);
        if (it_a != id_map.end() && it_b != id_map.end()) {
            new_between[pack_pair_key(it_a->second, it_b->second)] = kv.second;
        }
    }
    within_stats = std::move(new_within);
    between_stats = std::move(new_between);
}

/* --------------------------- Quadratic (trial) --------------------------- */

QuadraticStatsTrialCache::QuadraticStatsTrialCache(const QuadraticStatsCache& base_cache)
    : QuadraticStatsCache(base_cache.get_params()), base(base_cache) {
    // cached_lpdf, within_stats, between_stats all start empty/zero (default
    // member initializers in the base class) -- this trial tracks a DELTA,
    // not an absolute lpdf, and reads fall back to `base` (see below).
}

PairStat QuadraticStatsTrialCache::get_within(int k) const {
    auto it = within_stats.find(k);
    if (it != within_stats.end()) return it->second;
    return base.get_within(k);
}
PairStat QuadraticStatsTrialCache::get_between(int a, int b) const {
    auto it = between_stats.find(pack_pair_key(a, b));
    if (it != between_stats.end()) return it->second;
    return base.get_between(a, b);
}

void QuadraticStatsTrialCache::fold_into(QuadraticStatsCache& target) const {
    for (const auto& kv : within_stats) target.put_within(kv.first, kv.second);
    for (const auto& kv : between_stats) {
        auto [a, b] = unpack_pair_key(kv.first);
        target.put_between(a, b, kv.second);
    }
    target.adjust_lpdf(cached_lpdf);
}

/* ================================ Linear ================================ */

LinearStatsCache::LinearStatsCache(const LinearLikelihoodParams& p) : params(p) {
    const_within = params.prior_shape_within * std::log(params.prior_rate_within) - std::lgamma(params.prior_shape_within);
}

double LinearStatsCache::f_within(const PairStat& s) const {
    if (s.n_pairs <= 0) return 0.0;
    return const_within
         + (params.shape_within - 1.0) * s.sum_log_d
         + (-params.prior_shape_within - params.shape_within * s.n_pairs) * std::log(params.prior_rate_within + s.sum_d)
         - s.n_pairs * std::lgamma(params.shape_within)
         + std::lgamma(params.prior_shape_within + s.n_pairs * params.shape_within);
}

void LinearStatsCache::rebuild(const arma::mat& dist_matrix, const arma::uvec& allocs, const std::vector<arma::uword>& centres) {
    // PRECONDITION: `allocs` values are themselves centre POINT INDICES
    // (i.e. allocs(i) == the point-index of i's cluster's centre; a centre's
    // own entry is therefore allocs(c) == c). Callers must pass a
    // "cache_id_allocs"-style vector, never the positional 0..K-1 ids used
    // for compute_tessellation -- see VoronoiSampler::sync_cache_ids(). This
    // makes the within_stats key directly usable with no centre lookup.
    within_stats.clear();
    for (arma::uword c : centres) within_stats[static_cast<int>(c)] = PairStat{};
    for (arma::uword i = 0; i < allocs.n_elem; ++i) {
        arma::uword centre_idx = allocs(i);
        if (centre_idx == i) continue; // i is itself a centre: no self-pair
        double d = dist_matrix(centre_idx, i);
        if (d <= 0.0) continue;
        PairStat& s = within_stats[static_cast<int>(centre_idx)];
        s.n_pairs++; s.sum_d += d; s.sum_log_d += std::log(d);
    }
    cached_lpdf = 0.0;
    for (const auto& kv : within_stats) cached_lpdf += f_within(kv.second);
    recompute_between(dist_matrix, centres);
}

void LinearStatsCache::add_point(const arma::mat& dist_matrix, arma::uword point_idx, int cluster_id, arma::uword centre_idx) {
    if (point_idx == centre_idx) return;
    double d = dist_matrix(centre_idx, point_idx);
    if (d <= 0.0) return;
    PairStat s = within_stats.count(cluster_id) ? within_stats[cluster_id] : PairStat{};
    cached_lpdf -= f_within(s);
    s.n_pairs++; s.sum_d += d; s.sum_log_d += std::log(d);
    cached_lpdf += f_within(s);
    within_stats[cluster_id] = s;
}

void LinearStatsCache::remove_point(const arma::mat& dist_matrix, arma::uword point_idx, int cluster_id, arma::uword centre_idx) {
    if (point_idx == centre_idx) return;
    double d = dist_matrix(centre_idx, point_idx);
    if (d <= 0.0) return;
    PairStat s = within_stats.count(cluster_id) ? within_stats[cluster_id] : PairStat{};
    cached_lpdf -= f_within(s);
    s.n_pairs--; s.sum_d -= d; s.sum_log_d -= std::log(d);
    cached_lpdf += f_within(s);
    within_stats[cluster_id] = s;
}

void LinearStatsCache::recompute_between(const arma::mat& dist_matrix, const std::vector<arma::uword>& centres) {
    cached_lpdf -= between_lpdf;
    between_lpdf = 0.0;
    if (params.repulsion && centres.size() > 1) {
        double sum_d = 0.0, sum_log_d = 0.0;
        long n_pairs = 0;
        for (size_t a = 0; a < centres.size(); ++a) {
            for (size_t b = a + 1; b < centres.size(); ++b) {
                double d = dist_matrix(centres[a], centres[b]);
                if (d <= 0.0) continue;
                sum_d += d; sum_log_d += std::log(d); n_pairs++;
            }
        }
        if (n_pairs > 0) {
            between_lpdf = n_pairs * params.shape_between * std::log(params.rate_between)
                         - n_pairs * std::lgamma(params.shape_between)
                         - params.rate_between * sum_d
                         + (params.shape_between - 1.0) * sum_log_d;
        }
    }
    cached_lpdf += between_lpdf;
}
