"""
Phonetic decision tree for context-dependent state tying (paper Section V).

A triphone is a phone in the context of its left and right neighbour. There
are too many triphones to train a separate GMM for each, and most are rare or
unseen, so we cluster the HMM states of acoustically similar triphones to
share one GMM (a "tied state" or senone).

The clustering is a decision tree. For each (centre phone, HMM-state position)
we start with all observed left/right contexts pooled together and greedily
split them with yes/no questions about the left or right context phone,
each split chosen to maximise the data likelihood under a single diagonal
Gaussian. Splitting stops at a target number of leaves; each leaf is a senone.

Questions are generated automatically by clustering the phones on their
monophone means (Kaldi does the same), so no hand-written linguistic phone
classes are needed.
"""

import numpy as np

VAR_FLOOR = 0.01


def _two_means(keys, vecs, iters=10):
    """Split keys into 2 clusters by k-means on their vectors. Returns (setA, setB)."""
    if len(keys) < 2:
        return set(keys), set()
    c0 = vecs[0]
    d = np.sum((vecs - c0) ** 2, axis=1)
    a, b = c0.copy(), vecs[int(np.argmax(d))].copy()
    assign = np.zeros(len(keys), dtype=int)
    for it in range(iters):
        da = np.sum((vecs - a) ** 2, axis=1)
        db = np.sum((vecs - b) ** 2, axis=1)
        new = (db < da).astype(int)
        if it > 0 and np.array_equal(new, assign):
            break
        assign = new
        if np.any(assign == 0):
            a = vecs[assign == 0].mean(axis=0)
        if np.any(assign == 1):
            b = vecs[assign == 1].mean(axis=0)
    setA = {keys[i] for i in range(len(keys)) if assign[i] == 0}
    setB = {keys[i] for i in range(len(keys)) if assign[i] == 1}
    return setA, setB


def generate_questions(phone_means):
    """
    Auto-generate phone-set questions by recursively 2-means clustering the
    phones on their mean vectors. Each cluster encountered becomes a question
    ("is the context phone in this set?"). Singletons are included too.

    phone_means: dict phone_id -> mean vector.
    Returns: list of frozensets of phone ids.
    """
    questions = set()
    phones = sorted(phone_means.keys())
    for p in phones:
        questions.add(frozenset([p]))

    def recurse(keys):
        if len(keys) < 2:
            return
        vecs = np.array([phone_means[k] for k in keys])
        a, b = _two_means(keys, vecs)
        for s in (a, b):
            if 0 < len(s) < len(keys):
                questions.add(frozenset(s))
                recurse(sorted(s))

    recurse(phones)
    return [q for q in questions if len(q) > 0]


class _Node:
    __slots__ = ("contexts", "n", "s", "sq", "question", "side", "yes", "no", "pdf")

    def __init__(self, contexts, n, s, sq):
        self.contexts = contexts   # list of (left, right) tuples
        self.n = n
        self.s = s
        self.sq = sq
        self.question = None
        self.side = None           # 'L' or 'R'
        self.yes = None
        self.no = None
        self.pdf = None


def _objf(n, s, sq):
    """Single diagonal-Gaussian objective: -0.5 * sum_d n*log(var_d). Higher = tighter."""
    if n <= 0:
        return 0.0
    var = sq / n - (s / n) ** 2
    var = np.maximum(var, VAR_FLOOR)
    return -0.5 * n * float(np.sum(np.log(var)))


class DecisionTree:
    """Maps (centre phone, hmm-state position, left, right) -> tied pdf id."""

    def __init__(self, num_states_per_phone=3):
        self.roots = {}   # (centre, state_pos) -> _Node
        self.num_states_per_phone = num_states_per_phone
        self.num_pdfs = 0

    def map(self, centre, state_pos, left, right):
        node = self.roots.get((centre, state_pos))
        if node is None:
            return None
        while node.question is not None:
            ctx = left if node.side == "L" else right
            node = node.yes if ctx in node.question else node.no
        return node.pdf


def build_tree(stats, questions, max_leaves=200, min_count=200,
               num_states_per_phone=3):
    """
    Build the decision tree.

    stats: dict (centre, state_pos) -> dict (left, right) -> [n, s(D,), sq(D,)].
    questions: list of frozensets of phone ids.
    max_leaves: target total number of tied states (senones).
    min_count: both children of a split must have at least this many frames.

    Returns a DecisionTree.
    """
    tree = DecisionTree(num_states_per_phone)
    node_key = {}   # id(node) -> (centre, state_pos), to read per-context stats

    leaves = []
    for key, ctx_stats in stats.items():
        contexts = list(ctx_stats.keys())
        D = len(next(iter(ctx_stats.values()))[1])
        n = sum(ctx_stats[c][0] for c in contexts)
        s = np.zeros(D)
        sq = np.zeros(D)
        for c in contexts:
            s += ctx_stats[c][1]
            sq += ctx_stats[c][2]
        root = _Node(contexts, n, s, sq)
        tree.roots[key] = root
        node_key[id(root)] = key
        leaves.append(root)

    def best_split(node):
        if len(node.contexts) < 2:
            return None
        key = node_key[id(node)]
        ctx_stats = stats[key]
        parent_obj = _objf(node.n, node.s, node.sq)
        best = None
        for side in ("L", "R"):
            idx = 0 if side == "L" else 1
            for q in questions:
                yn = ys = ysq = None
                nn = ns = nsq = None
                yc, nc = [], []
                for c in node.contexts:
                    st = ctx_stats[c]
                    if c[idx] in q:
                        yc.append(c)
                        if yn is None:
                            yn, ys, ysq = 0.0, np.zeros_like(st[1]), np.zeros_like(st[2])
                        yn += st[0]; ys = ys + st[1]; ysq = ysq + st[2]
                    else:
                        nc.append(c)
                        if nn is None:
                            nn, ns, nsq = 0.0, np.zeros_like(st[1]), np.zeros_like(st[2])
                        nn += st[0]; ns = ns + st[1]; nsq = nsq + st[2]
                if not yc or not nc or yn < min_count or nn < min_count:
                    continue
                gain = _objf(yn, ys, ysq) + _objf(nn, ns, nsq) - parent_obj
                if best is None or gain > best[0]:
                    best = (gain, side, q, _Node(yc, yn, ys, ysq), _Node(nc, nn, ns, nsq))
        return best

    splits = {id(node): best_split(node) for node in leaves}

    while len(leaves) < max_leaves:
        cand_i = -1
        cand = None
        for i, node in enumerate(leaves):
            sp = splits.get(id(node))
            if sp is None:
                continue
            if cand is None or sp[0] > cand[0]:
                cand, cand_i = sp, i
        if cand is None or cand[0] <= 0:
            break
        gain, side, q, yes, no = cand
        node = leaves[cand_i]
        node.question, node.side, node.yes, node.no = q, side, yes, no
        key = node_key[id(node)]
        node_key[id(yes)] = key
        node_key[id(no)] = key
        leaves.pop(cand_i)
        del splits[id(node)]
        for child in (yes, no):
            leaves.append(child)
            splits[id(child)] = best_split(child)

    for pdf, node in enumerate(leaves):
        node.pdf = pdf
    tree.num_pdfs = len(leaves)
    return tree
