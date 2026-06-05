"""
3-state left-to-right HMM topology (paper Section IV-C).

Each phone is modeled as:
  S1 ──▶ S2 ──▶ S3 ──▶ (exit)
  with self-loops at each state for variable duration.

Transitions are stored as log-probabilities (negative log space: lower = more probable).

The HMM topology is used to build the H FST in the decoder.
"""

from typing import List, Tuple

# Log-probability constants (transition weights)
# These are typical values for monophone HMMs.
# In a real system they'd be learned from data.
SELF_LOOP_LOG_PROB = -0.5       # log(0.6) roughly: stay in same state
TRANSITION_LOG_PROB = -0.9      # log(0.4) roughly: move to next state
EXIT_LOG_PROB = -0.9            # log(0.4): exit from final state


def build_phone_hmm(
    phone_id: int,
    pdf_offset: int,
    self_loop: float = SELF_LOOP_LOG_PROB,
    forward: float = TRANSITION_LOG_PROB,
    exit_prob: float = EXIT_LOG_PROB,
) -> dict:
    """
    Build a 3-state left-to-right HMM for one phone.

    Each state has a unique pdf-id (the GMM index for that state).
    pdf_offset is the starting pdf-id for this phone's states.

    Returns a dict:
        {
            "phone_id": int,
            "states": [  # 3 entries, one per HMM state
                {"self_loop_logp": float, "forward_logp": float, "pdf_id": int},
                ...
            ],
            "exit_logp": float,  # probability of leaving the final state
        }
    """
    return {
        "phone_id": phone_id,
        "states": [
            {"self_loop_logp": self_loop, "forward_logp": forward, "pdf_id": pdf_offset + 0},
            {"self_loop_logp": self_loop, "forward_logp": forward, "pdf_id": pdf_offset + 1},
            {"self_loop_logp": self_loop, "forward_logp": forward, "pdf_id": pdf_offset + 2},
        ],
        "exit_logp": exit_prob,
    }


def build_utterance_hmm(
    phone_ids: List[int],
    phone_hmms: List[dict],
) -> Tuple[List[dict], int]:
    """
    Build a concatenated HMM for a known phone sequence (e.g., "zero" = z iy r ow).

    Glues each phone's HMM end-to-end: the exit of phone N connects to
    the first state of phone N+1.

    Args:
        phone_ids: ordered list of phone IDs that make up the utterance.
        phone_hmms: list of all phone HMMs (from build_phone_hmm).

    Returns:
        (states_list, num_pdfs):
            states_list: list of dicts, one per HMM state in the concatenated graph,
                each with keys: pdf_id, transition_logp (to next state), self_loop_logp.
            num_pdfs: total number of unique pdf-ids covered.
    """
    hmm_map = {h["phone_id"]: h for h in phone_hmms}
    states = []
    for pid in phone_ids:
        hmm = hmm_map[pid]
        for s in hmm["states"]:
            states.append(s)
    return states, max(s["pdf_id"] for s in states) + 1


def total_pdfs(phone_hmms: List[dict]) -> int:
    """Return the total number of pdf-ids across all phones (3 per phone)."""
    if not phone_hmms:
        return 0
    last = phone_hmms[-1]
    last_state = last["states"][-1]
    return last_state["pdf_id"] + 1


def build_all_phone_hmms(num_phones: int) -> List[dict]:
    """
    Build HMMs for all phones with consecutively numbered pdf-ids.
    Phone 0 gets pdf-ids 0, 1, 2; phone 1 gets 3, 4, 5; etc.
    """
    hmms = []
    for pid in range(num_phones):
        hmms.append(build_phone_hmm(pid, pdf_offset=pid * 3))
    return hmms


if __name__ == "__main__":
    # Test
    hmms = build_all_phone_hmms(5)
    print(f"Built {len(hmms)} phone HMMs")
    print(f"  Total pdf-ids: {total_pdfs(hmms)}")
    print(f"  Phone 0 pdf-ids: {[s['pdf_id'] for s in hmms[0]['states']]}")
    print(f"  Phone 1 pdf-ids: {[s['pdf_id'] for s in hmms[1]['states']]}")

    # Test utterance HMM
    states, npdf = build_utterance_hmm([0, 1, 2], hmms)
    print(f"\nUtterance [0,1,2]:")
    print(f"  {len(states)} total HMM states, {npdf} pdf-ids")
    print(f"  State pdf_ids: {[s['pdf_id'] for s in states]}")
