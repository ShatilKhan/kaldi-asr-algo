"""
Weighted Finite-State Transducer (WFST) for ASR decoding (paper Sections II, VII).

An FST is a graph with states and arcs. Each arc has:
  - input label (symbol consumed)
  - output label (symbol produced)
  - weight (cost, lower = better)

Key operations:
  - compose(A, B): glue two FSTs together at matching symbols (Section VII)
  - best_path(fst, input_ids): Viterbi search over fst given input symbols
  - Shortest path on the composed graph = recognized word sequence

Epsilon (ε) labels are represented as -1.
"""

from typing import List, Tuple, Optional
import heapq

EPS = -1  # epsilon label


class Arc:
    """One arc in an FST."""

    __slots__ = ("next_state", "ilabel", "olabel", "weight")

    def __init__(self, next_state: int, ilabel: int, olabel: int, weight: float):
        self.next_state = next_state
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight

    def __repr__(self) -> str:
        i = f"ε" if self.ilabel == EPS else str(self.ilabel)
        o = f"ε" if self.olabel == EPS else str(self.olabel)
        return f"Arc({i}:{o}/{self.weight:.2f} → {self.next_state})"


class FST:
    """
    Weighted Finite-State Transducer.

    States are consecutive integers (0, 1, 2, ...).
    Arc labels are integers; EPS (-1) represents epsilon.
    Weights are additive costs (tropical semiring: lower = better).
    """

    def __init__(self):
        # List of arcs per state: arcs[state] = [Arc, ...]
        self.arcs: List[List[Arc]] = []
        self.start_state: int = 0
        self.final_states: set = set()
        self.final_weights: dict = {}  # state -> weight

    def add_state(self) -> int:
        """Add a new state, return its index."""
        s = len(self.arcs)
        self.arcs.append([])
        return s

    def add_arc(self, from_state: int, arc: Arc) -> None:
        """Add an arc from from_state."""
        while from_state >= len(self.arcs):
            self.arcs.append([])
        self.arcs[from_state].append(arc)

    def set_start(self, state: int) -> None:
        self.start_state = state

    def set_final(self, state: int, weight: float = 0.0) -> None:
        self.final_states.add(state)
        self.final_weights[state] = weight

    @property
    def num_states(self) -> int:
        return len(self.arcs)

    @property
    def num_arcs(self) -> int:
        return sum(len(a) for a in self.arcs)

    def num_eps_arcs(self) -> int:
        """Count arcs with epsilon on either label."""
        return sum(
            1 for arcs in self.arcs for a in arcs if a.ilabel == EPS or a.olabel == EPS
        )

    def print_stats(self) -> str:
        """Return a string with basic stats."""
        return f"FST({self.num_states} states, {self.num_arcs} arcs, {self.num_eps_arcs()} ε-arcs)"

    def __repr__(self) -> str:
        return self.print_stats()

    def copy(self) -> "FST":
        """Create a deep copy of this FST."""
        fst = FST()
        for _ in range(self.num_states):
            fst.add_state()
        for s in range(self.num_states):
            for a in self.arcs[s]:
                fst.add_arc(s, Arc(a.next_state, a.ilabel, a.olabel, a.weight))
        fst.set_start(self.start_state)
        for s in self.final_states:
            fst.set_final(s, self.final_weights.get(s, 0.0))
        return fst


# ---- Composition ----

# Epsilon filter states for composition
FILTER_NORMAL = 0   # both FSTs consume/produce normally
FILTER_EPS_A = 1    # FST A has epsilon output: advance A without consuming B input
FILTER_EPS_B = 2    # FST B has epsilon input: advance B without consuming A output


def compose(a: FST, b: FST) -> FST:
    """
    Compose two WFSTs with an epsilon filter (paper Section VII).

    A ∘ B: match output labels of A with input labels of B.
    Uses the standard 3-state epsilon filter to handle epsilon arcs.

    Args:
        a: First FST (outputs are matched).
        b: Second FST (inputs are matched).

    Returns:
        Composed FST. Input labels = A's input, Output labels = B's output.
    """
    result = FST()

    # Composition state = (a_state, b_state, filter_state)
    # We'll build a mapping from (a, b, f) -> result_state
    state_map: dict = {}
    start = (a.start_state, b.start_state, FILTER_NORMAL)
    s0 = result.add_state()
    state_map[start] = s0
    result.set_start(s0)

    # Priority queue for BFS
    queue = [start]
    visited = {start}

    while queue:
        sa, sb, sf = queue.pop(0)
        rs = state_map[(sa, sb, sf)]

        # Check if this is a final state
        a_is_final = sa in a.final_states
        b_is_final = sb in b.final_states
        if a_is_final and b_is_final and sf == FILTER_NORMAL:
            result.set_final(
                rs, a.final_weights.get(sa, 0.0) + b.final_weights.get(sb, 0.0)
            )

        # Process arcs based on filter state
        if sf == FILTER_NORMAL:
            # Normal: match A's output with B's input
            # First, handle epsilon on A's output
            for a_arc in a.arcs[sa]:
                if a_arc.olabel == EPS:
                    key = (a_arc.next_state, sb, FILTER_EPS_A)
                    if key not in state_map:
                        ns = result.add_state()
                        state_map[key] = ns
                    if key not in visited:
                        visited.add(key)
                        queue.append(key)
                    result.add_arc(rs, Arc(state_map[key], a_arc.ilabel, EPS, a_arc.weight))
                else:
                    # Match with B's arcs that have this label on input
                    for b_arc in b.arcs[sb]:
                        if b_arc.ilabel == a_arc.olabel:
                            key = (a_arc.next_state, b_arc.next_state, FILTER_NORMAL)
                            if key not in state_map:
                                ns = result.add_state()
                                state_map[key] = ns
                            if key not in visited:
                                visited.add(key)
                                queue.append(key)
                            result.add_arc(
                                rs,
                                Arc(
                                    state_map[key],
                                    a_arc.ilabel,
                                    b_arc.olabel,
                                    a_arc.weight + b_arc.weight,
                                ),
                            )

        elif sf == FILTER_EPS_A:
            # A just output epsilon. We are at (sa, sb) where A advanced via epsilon.
            # Two cases:
            # 1. B has epsilon-input arcs: consume them and go to NORMAL
            for b_arc in b.arcs[sb]:
                if b_arc.ilabel == EPS:
                    key = (sa, b_arc.next_state, FILTER_NORMAL)
                    if key not in state_map:
                        ns = result.add_state()
                        state_map[key] = ns
                    if key not in visited:
                        visited.add(key)
                        queue.append(key)
                    result.add_arc(rs, Arc(state_map[key], EPS, b_arc.olabel, b_arc.weight))

            # 2. Try matching A's non-epsilon outputs with B's inputs → back to NORMAL
            for a_arc in a.arcs[sa]:
                if a_arc.olabel == EPS:
                    # A still has epsilon outputs — stay in EPS_A
                    key = (a_arc.next_state, sb, FILTER_EPS_A)
                    if key not in state_map:
                        ns = result.add_state()
                        state_map[key] = ns
                    if key not in visited:
                        visited.add(key)
                        queue.append(key)
                    result.add_arc(rs, Arc(state_map[key], a_arc.ilabel, EPS, a_arc.weight))
                else:
                    # A has a non-epsilon output — try to match with B's input
                    for b_arc in b.arcs[sb]:
                        if b_arc.ilabel == a_arc.olabel:
                            key = (a_arc.next_state, b_arc.next_state, FILTER_NORMAL)
                            if key not in state_map:
                                ns = result.add_state()
                                state_map[key] = ns
                            if key not in visited:
                                visited.add(key)
                                queue.append(key)
                            result.add_arc(
                                rs,
                                Arc(
                                    state_map[key],
                                    a_arc.ilabel,
                                    b_arc.olabel,
                                    a_arc.weight + b_arc.weight,
                                ),
                            )

        elif sf == FILTER_EPS_B:
            # B just consumed epsilon from A, process A's epsilon-output arcs to match
            for a_arc in a.arcs[sa]:
                if a_arc.olabel == EPS:
                    key = (a_arc.next_state, sb, FILTER_EPS_B)
                    if key not in state_map:
                        ns = result.add_state()
                        state_map[key] = ns
                    if key not in visited:
                        visited.add(key)
                        queue.append(key)
                    result.add_arc(rs, Arc(state_map[key], a_arc.ilabel, EPS, a_arc.weight))
                else:
                    # Match with B's input labels
                    for b_arc in b.arcs[sb]:
                        if b_arc.ilabel == a_arc.olabel:
                            key = (a_arc.next_state, b_arc.next_state, FILTER_NORMAL)
                            if key not in state_map:
                                ns = result.add_state()
                                state_map[key] = ns
                            if key not in visited:
                                visited.add(key)
                                queue.append(key)
                            result.add_arc(
                                rs,
                                Arc(
                                    state_map[key],
                                    a_arc.ilabel,
                                    b_arc.olabel,
                                    a_arc.weight + b_arc.weight,
                                ),
                            )

    return result


# ---- Viterbi best-path ----

def best_path(fst: FST, input_ids: List[int]) -> Tuple[List[int], float]:
    """
    Find the best path through the FST given a sequence of input symbols.

    Uses token-passing Viterbi (paper Section VIII):
      - At each input symbol, propagate tokens along matching arcs
      - Merge tokens on the same state, keep the best
      - Backpointers for path recovery

    Args:
        fst: The FST to search.
        input_ids: Sequence of input symbols to match.

    Returns:
        (output_sequence, total_cost)
        output_sequence: list of output symbols from the best path.
        total_cost: cumulative cost of the best path.
    """
    # Token: (score, state, history_of_output_symbols)
    # We'll track best score to each state at current position.
    tokens = {fst.start_state: (0.0, [])}

    for inp in input_ids:
        new_tokens = {}
        for state, (score, out_seq) in tokens.items():
            for arc in fst.arcs[state]:
                # Try epsilon input arc (consumes no input)
                if arc.ilabel == EPS:
                    new_score = score + arc.weight
                    new_out = out_seq + ([arc.olabel] if arc.olabel != EPS else [])
                    if arc.next_state not in new_tokens or new_score < new_tokens[arc.next_state][0]:
                        new_tokens[arc.next_state] = (new_score, new_out)

                # Try matching input label
                elif arc.ilabel == inp:
                    new_score = score + arc.weight
                    new_out = out_seq + ([arc.olabel] if arc.olabel != EPS else [])
                    if arc.next_state not in new_tokens or new_score < new_tokens[arc.next_state][0]:
                        new_tokens[arc.next_state] = (new_score, new_out)

        # After the main matching, propagate epsilon arcs forward (ε-closure)
        changed = True
        while changed:
            changed = False
            for state, (score, out_seq) in list(new_tokens.items()):
                for arc in fst.arcs[state]:
                    if arc.ilabel == EPS:
                        new_score = score + arc.weight
                        new_out = out_seq + ([arc.olabel] if arc.olabel != EPS else [])
                        if arc.next_state not in new_tokens or new_score < new_tokens[arc.next_state][0]:
                            new_tokens[arc.next_state] = (new_score, new_out)
                            changed = True

        tokens = new_tokens
        if not tokens:
            # All paths died — no path through the FST for this input
            return ([], float("inf"))

    # At the end, pick the best final state
    best_score = float("inf")
    best_out = []
    for state, (score, out_seq) in tokens.items():
        if state in fst.final_states:
            final_score = score + fst.final_weights.get(state, 0.0)
            if final_score < best_score:
                best_score = final_score
                best_out = out_seq

    # If no path ended in a final state, pick the best non-final
    if best_score == float("inf"):
        for state, (score, out_seq) in tokens.items():
            if score < best_score:
                best_score = score
                best_out = out_seq

    return (best_out, best_score)


# ---- Linear chain FST builder (for H, L, G) ----

def linear_chain_fst(
    input_labels: List[int],
    output_labels: List[int],
    weights: Optional[List[float]] = None,
    start_weight: float = 0.0,
    final_weight: float = 0.0,
) -> FST:
    """
    Build a linear chain FST: s0 --in0:out0/w0--> s1 --in1:out1/w1--> ... --/> final.

    Each arc moves to the next state. Used for building phone sequences in the L FST.

    Args:
        input_labels: list of input symbols.
        output_labels: list of output symbols (same length, or one shorter with final output).
        weights: arc weights (default 0.0).
        start_weight: weight on the initial state (default 0.0).
        final_weight: weight on the final state (default 0.0).

    Returns:
        FST with len(input_labels) arcs.
    """
    fst = FST()
    s0 = fst.add_state()
    fst.set_start(s0)

    if weights is None:
        weights = [0.0] * len(input_labels)

    prev = s0
    for i in range(len(input_labels)):
        ns = fst.add_state()
        ilabel = input_labels[i]
        olabel = output_labels[i] if i < len(output_labels) else EPS
        w = weights[i] if i < len(weights) else 0.0
        fst.add_arc(prev, Arc(ns, ilabel, olabel, w))
        prev = ns

    fst.set_final(prev, final_weight)
    return fst


def pretty_print_fst(fst: FST, isymbols: dict = None, osymbols: dict = None) -> str:
    """
    Pretty-print an FST with optional symbol tables.

    Args:
        fst: The FST to print.
        isymbols: mapping from int label -> str (default identity).
        osymbols: mapping from int label -> str (default identity).

    Returns:
        Multi-line string representation.
    """
    if isymbols is None:
        isymbols = {EPS: "ε"}
    if osymbols is None:
        osymbols = {EPS: "ε"}

    lines = []
    lines.append(f"FST: {fst.num_states} states, {fst.num_arcs} arcs")
    for s in range(fst.num_states):
        start_mark = " →" if s == fst.start_state else "  "
        final_mark = " ◇" if s in fst.final_states else "  "
        lines.append(f"{start_mark}State {s}{final_mark}")
        for a in fst.arcs[s]:
            il = isymbols.get(a.ilabel, str(a.ilabel))
            ol = osymbols.get(a.olabel, str(a.olabel))
            lines.append(f"     {il}:{ol}/{a.weight:.3f} → {a.next_state}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test: compose two simple FSTs
    # A maps a→x, b→y
    # B maps x→1, y→2
    # A∘B should map a→1, b→2

    a = FST()
    s0 = a.add_state()
    a.set_start(s0)
    s1 = a.add_state()
    a.set_final(s1)
    a.add_arc(s0, Arc(s1, 0, 10, 0.5))  # a → x
    a.add_arc(s0, Arc(s1, 1, 20, 0.3))  # b → y

    b = FST()
    s0 = b.add_state()
    b.set_start(s0)
    s1 = b.add_state()
    b.set_final(s1)
    b.add_arc(s0, Arc(s1, 10, 100, 0.2))  # x → 1
    b.add_arc(s0, Arc(s1, 20, 200, 0.4))  # y → 2

    c = compose(a, b)
    print("Composed FST A∘B:")
    print(pretty_print_fst(c))
    print()

    # Test best_path
    out_a, cost_a = best_path(c, [0])  # input = a, should output 1
    out_b, cost_b = best_path(c, [1])  # input = b, should output 2
    print(f"best_path(c, [a]) → outputs {out_a}, cost={cost_a:.1f} (expect [100], cost=0.7)")
    print(f"best_path(c, [b]) → outputs {out_b}, cost={cost_b:.1f} (expect [200], cost=0.7)")
