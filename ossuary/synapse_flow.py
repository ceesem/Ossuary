import numpy as np


def covering_paths(sk):
    cov_paths = []
    seen = np.full(len(sk.vertices), False)
    ep_order = np.argsort(sk.distance_to_root[sk.end_points])[::-1]
    for ep in sk.end_points[ep_order]:
        ptr = np.array(sk.path_to_root(ep))
        cov_paths.append(ptr[~seen[ptr]])
        seen[ptr] = True
    return cov_paths

def distribution_split_entropy( counts ):
    if np.sum(counts)==0:
        return 0
    ps = np.divide(counts, np.sum(counts, axis=1)[:,np.newaxis], where=np.sum(counts, axis=1)[:,np.newaxis]>0)
    Hpart = np.sum(np.multiply(ps, np.log2(ps, where=ps>0)), axis=1)
    Hws = np.sum(counts, axis=1) / np.sum(counts)
    Htot = -np.sum(Hpart * Hws)
    return Htot

def synapse_betweenness(sk, pre_inds, post_inds, use_entropy=False):
    def _precompute_synapse_inds(syn_inds):
        Nsyn = len(syn_inds)
        n_syn = np.zeros(len(sk.vertices))
        for ind in syn_inds:
            n_syn[ind] += 1
        return Nsyn, n_syn
    
    Npre, n_pre = _precompute_synapse_inds(pre_inds)
    Npost, n_post = _precompute_synapse_inds(post_inds)
    
    syn_btwn = np.zeros(len(sk.vertices))
    split_index = np.zeros(len(sk.vertices))
    cov_paths_rev = covering_paths(sk)[::-1]
    if use_entropy:
        entropy_normalization = distribution_split_entropy(np.array([[Npre,Npost],[0,0]]))
    for path in cov_paths_rev:
        downstream_pre = 0
        downstream_post = 0
        for ind in path:
            downstream_pre += n_pre[ind]
            downstream_post += n_post[ind]
            syn_btwn[ind] = downstream_pre * (Npost - downstream_post) + \
                               (Npre - downstream_pre) * downstream_post
            if use_entropy:
                counts = np.array([[downstream_pre, downstream_post],
                                   [Npost - downstream_post, Npre - downstream_pre]])
                split_index[ind] = 1 - distribution_split_entropy(counts)/entropy_normalization
        # Deposit each branch's synapses at the branch point.
        bp_ind = sk.parent_node(path[-1])
        if bp_ind is not None:
            n_pre[bp_ind]  += downstream_pre
            n_post[bp_ind] += downstream_post
    if use_entropy:
        return syn_btwn, split_index
    else:
        return syn_btwn

def find_axon_split(sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True):
    syn_btwn = synapse_betweenness(sk, pre_inds, post_inds)
    high_vinds = np.flatnonzero(syn_btwn==max(syn_btwn))
    close_vind = high_vinds[np.argmin(sk.distance_to_root[high_vinds])]
    
    if return_quality:
        axon_qual_label = np.full(len(sk.vertices), False)
        axon_qual_label[ sk.downstream_nodes(close_vind) ] = True
        split_quality = axon_split_quality(axon_qual_label, pre_inds, post_inds)

    if extend_to_segment:
        relseg = sk.segment_map[close_vind]
        axon_split_ind = sk.segments[relseg][-1]
    else:
        axon_split_ind = close_vind
    
    if return_quality:
        return axon_split_ind, split_quality
    else:
        return axon_split_ind

def split_axon_by_synapses(sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True):
    axon_split = find_axon_split(sk, pre_inds, post_inds, return_quality=return_quality, extend_to_segment=True)
    if return_quality:
        axon_split_ind, split_quality = axon_split
    else:
        axon_split_ind = axon_split
    is_axon = np.full(len(sk.vertices), False)
    is_axon[ sk.downstream_nodes(axon_split_ind) ] = True
    
    if return_quality:
        return is_axon, split_quality
    else:
        return is_axon

def axon_split_quality(is_axon, pre_inds, post_inds):
    axon_pre = sum(is_axon[pre_inds])
    axon_post = sum(is_axon[post_inds])
    dend_pre = sum(~is_axon[pre_inds])
    dend_post = sum(~is_axon[post_inds])

    counts = np.array([[axon_pre, axon_post],[dend_pre, dend_post]])
    observed_ent = distribution_split_entropy( counts )
    
    unsplit_ent = distribution_split_entropy([[len(pre_inds), len(post_inds)]] )

    return 1-observed_ent/unsplit_ent