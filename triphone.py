"""
Context-dependent (triphone) acoustic modelling (paper Section V).

Bootstraps from a trained monophone model:
  1. force-align the training audio with the monophones, recording each
     frame's (centre phone, HMM-state position, left phone, right phone),
  2. accumulate single-Gaussian stats per context and build a phonetic
     decision tree (tree.py) that ties the triphone states into senones,
  3. initialise one GMM per senone from its leaf stats and refine with a few
     align/re-estimate iterations, splitting up to multiple Gaussians,
  4. build a context-dependent decoding graph whose input labels are senones.

Scope: word-internal triphones (word-boundary context = silence). Cross-word
context needs a C transducer + determinization, which this toy doesn't build.
"""

import numpy as np

from fst import FST, Arc, EPS
from gmm import DiagGmm, train_gmm
from hmm import (build_all_phone_hmms, build_utterance_hmm,
                 SELF_LOOP_LOG_PROB, TRANSITION_LOG_PROB, EXIT_LOG_PROB)
from tree import generate_questions, build_tree
from train import build_phone_sequence, reestimate_gmms, EM_ITERS

VAR_FLOOR = 1e-3


# --------------------------------------------------------------------------
# Forced alignment that returns the state path (not just pdf-ids)
# --------------------------------------------------------------------------

def _align_state_path(frames, states, gmms, acoustic_scale=1.0,
                      transition_scale=1.0, self_loop_scale=1.0):
    """Viterbi forced alignment; returns a per-frame state index into `states`."""
    num_frames = frames.shape[0]
    if num_frames == 0:
        return np.array([], dtype=int)
    num_states = len(states)
    log_likes = DiagGmm.score_batch_all(gmms, frames)
    self_loop = np.array([s["self_loop_logp"] for s in states]) * self_loop_scale
    forward = np.array([s["forward_logp"] for s in states]) * transition_scale
    pdf_ids = np.array([s["pdf_id"] for s in states])

    dp = np.full((num_frames, num_states), -1e30)
    back = np.zeros((num_frames, num_states), dtype=int)
    dp[0, 0] = log_likes[0, pdf_ids[0]] * acoustic_scale
    for t in range(1, num_frames):
        dp_self = dp[t - 1] + self_loop
        dp_fwd = np.full(num_states, -1e30)
        dp_fwd[1:] = dp[t - 1, :-1] + forward[:-1]
        stacked = np.column_stack([dp_self, dp_fwd])
        best = np.max(stacked, axis=1)
        back[t] = np.argmax(stacked, axis=1)
        dp[t] = best + log_likes[t, pdf_ids] * acoustic_scale

    final = num_states - 1
    if dp[-1, final] <= -1e29:
        final = int(np.argmax(dp[-1]))
    path = np.zeros(num_frames, dtype=int)
    path[-1] = final
    cur = final
    for t in range(num_frames - 2, -1, -1):
        cur = cur - back[t + 1, cur]
        path[t] = cur
    return path


# --------------------------------------------------------------------------
# Triphone state chains and stat accumulation
# --------------------------------------------------------------------------

def _contexts(phone_ids, sil):
    """For each (position, state_pos) yield (centre, state_pos, left, right)."""
    out = []
    n = len(phone_ids)
    for pos, centre in enumerate(phone_ids):
        left = phone_ids[pos - 1] if pos > 0 else sil
        right = phone_ids[pos + 1] if pos < n - 1 else sil
        for sp in range(3):
            out.append((centre, sp, left, right))
    return out


def accumulate_stats(frames_list, phone_seqs, phone_hmms, mono_gmms, sil):
    """
    Force-align each utterance with the monophone model and accumulate
    per-context single-Gaussian stats.

    Returns stats: (centre, state_pos) -> {(left, right): [n, sum(D,), sumsq(D,)]}.
    """
    stats = {}
    for frames, phones in zip(frames_list, phone_seqs):
        if frames.shape[0] == 0:
            continue
        states, _ = build_utterance_hmm(phones, phone_hmms)
        path = _align_state_path(frames, states, mono_gmms)
        ctxs = _contexts(phones, sil)
        for t, st_idx in enumerate(path):
            centre, sp, left, right = ctxs[st_idx]
            key = (centre, sp)
            d = stats.setdefault(key, {})
            entry = d.get((left, right))
            x = frames[t]
            if entry is None:
                d[(left, right)] = [1.0, x.copy(), x * x]
            else:
                entry[0] += 1.0
                entry[1] += x
                entry[2] += x * x
    return stats


def _phone_means(mono_gmms, num_phones):
    """Phone -> mean vector (averaged over its 3 monophone states, 1st component)."""
    means = {}
    for p in range(num_phones):
        vs = [mono_gmms[p * 3 + sp].means[0] for sp in range(3)]
        means[p] = np.mean(vs, axis=0)
    return means


def triphone_states(phone_ids, tree, sil):
    """Build the per-state chain for a phone sequence, pdf = tied senone."""
    states = []
    ctxs = _contexts(phone_ids, sil)
    for (centre, sp, left, right) in ctxs:
        pdf = tree.map(centre, sp, left, right)
        if pdf is None:
            pdf = 0  # unseen centre/state: fall back (rare)
        states.append({
            "pdf_id": pdf,
            "self_loop_logp": SELF_LOOP_LOG_PROB,
            "forward_logp": TRANSITION_LOG_PROB,
        })
    return states


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_triphone(transcripts, frames_list, lex, mono_gmms,
                   max_leaves=200, min_count=300, component_levels=(1, 2, 4),
                   iters_per_level=5, sil_between=True, verbose=True):
    """
    Train a context-dependent (triphone) model from a monophone bootstrap.
    Returns (tree, senone_gmms).
    """
    phone_hmms = build_all_phone_hmms(lex.num_phones)
    sil = lex.sil_phone
    phone_seqs = [build_phone_sequence(t, lex, sil_between=sil_between) for t in transcripts]

    if verbose:
        print("  [tri] accumulating triphone stats from monophone alignment...", flush=True)
    stats = accumulate_stats(frames_list, phone_seqs, phone_hmms, mono_gmms, sil)

    if verbose:
        nctx = sum(len(d) for d in stats.values())
        print(f"  [tri] {len(stats)} (phone,state) roots, {nctx} distinct contexts", flush=True)

    questions = generate_questions(_phone_means(mono_gmms, lex.num_phones))
    tree = build_tree(stats, questions, max_leaves=max_leaves, min_count=min_count)
    if verbose:
        print(f"  [tri] built tree: {len(questions)} questions -> {tree.num_pdfs} tied senones", flush=True)

    # Initialise one Gaussian per senone from its leaf pooled stats.
    leaf_stats = {}  # pdf -> [n, s, sq]
    for key, ctx_stats in stats.items():
        for (left, right), (n, s, sq) in ctx_stats.items():
            pdf = tree.map(key[0], key[1], left, right)
            if pdf is None:
                continue
            acc = leaf_stats.get(pdf)
            if acc is None:
                leaf_stats[pdf] = [n, s.copy(), sq.copy()]
            else:
                acc[0] += n; acc[1] += s; acc[2] += sq
    D = frames_list[0].shape[1]
    senone_gmms = []
    for pdf in range(tree.num_pdfs):
        acc = leaf_stats.get(pdf)
        if acc is None or acc[0] < 1:
            senone_gmms.append(DiagGmm(np.zeros((1, D)), np.ones((1, D)), np.ones(1)))
            continue
        n, s, sq = acc
        mean = s / n
        var = np.maximum(sq / n - mean ** 2, VAR_FLOOR)
        senone_gmms.append(DiagGmm(mean.reshape(1, -1), var.reshape(1, -1), np.ones(1)))

    # Refine: align with the triphone model, re-estimate, split up to K.
    for level_idx, n_comp in enumerate(component_levels):
        if verbose:
            print(f"  [tri] level {level_idx + 1}: {n_comp} Gaussians", flush=True)
        if level_idx > 0:
            senone_gmms = [g.split() for g in senone_gmms]
        for it in range(iters_per_level):
            alignments = []
            for frames, phones in zip(frames_list, phone_seqs):
                if frames.shape[0] == 0:
                    alignments.append(np.array([], dtype=int))
                    continue
                states = triphone_states(phones, tree, sil)
                path = _align_state_path(frames, states, senone_gmms)
                pdfs = np.array([states[i]["pdf_id"] for i in path])
                alignments.append(pdfs)
            senone_gmms = reestimate_gmms(alignments, frames_list, senone_gmms, n_comp)

    if verbose:
        print(f"  [tri] done: {tree.num_pdfs} senones, "
              f"{sum(g.K for g in senone_gmms)} total Gaussians", flush=True)
    return tree, senone_gmms


# --------------------------------------------------------------------------
# Context-dependent decoding graph (HL with senone input labels)
# --------------------------------------------------------------------------

def build_triphone_hl(lex, tree) -> FST:
    """
    Build the HL graph for triphones: a word loop where each word expands to
    its phone sequence, each phone to a 3-state left-to-right HMM whose input
    labels are the tied senone pdf-ids for that phone's within-word context.
    The word id is emitted on the exit arc. Optional silence loop. Compose
    with G (word loop) exactly like the monophone HCLG.

    Topology mirrors the monophone H exactly, inlined so no separate context
    transducer is needed: enter state 0 via epsilon, each state has a
    self-loop and a forward arc on its own pdf, the last state's exit arc
    returns to the loop and carries the word label.
    """
    sil = lex.sil_phone
    g = FST()
    start = g.add_state()
    g.set_start(start)
    g.set_final(start)

    sl = -SELF_LOOP_LOG_PROB
    fw = -TRANSITION_LOG_PROB
    ex = -EXIT_LOG_PROB

    def add_word(phone_ids, olabel_last):
        states = triphone_states(phone_ids, tree, sil)  # 3 per phone
        nodes = [g.add_state() for _ in states]
        g.add_arc(start, Arc(nodes[0], EPS, EPS, 0.0))  # enter first state (no frame)
        for i, stinfo in enumerate(states):
            pdf = stinfo["pdf_id"]
            node = nodes[i]
            g.add_arc(node, Arc(node, pdf, EPS, sl))    # self-loop
            if i < len(states) - 1:
                g.add_arc(node, Arc(nodes[i + 1], pdf, EPS, fw))  # forward
            else:
                g.add_arc(node, Arc(start, pdf, olabel_last, ex))  # exit, emit word

    add_word([sil], EPS)                                 # optional silence
    for word in lex.words:
        add_word(lex.lexicon[word], lex.word_map[word])
    return g
