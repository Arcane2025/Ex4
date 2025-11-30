"""
Option 3 - Creating a PPMI matrix

A. We define 20 football analytics phrases.
   1. For each phrase we draw a frequency between 20 and 100.
   2. For every pair of phrases we draw a co-occurrence count between
      0 and min(freq_i, freq_j).
B. We compute the PPMI matrix:
       PPMI(i,j) = max( log2( p(i,j) / (p(i) * p(j)) ), 0 )
   where p(i,j) and p(i) are estimated from the co-occurrence counts.
C. We display:
   - The frequencies
   - The co-occurrence matrix
   - The PPMI matrix
   - A heatmap visualization of the PPMI matrix
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def build_counts(random_seed: int = 123):
    phrases = [
        "counter attack", "high press", "xg model", "deep block",
        "false nine", "box midfield", "set piece", "through ball",
        "pressing trap", "low block", "overlapping run", "inverted fullback",
        "expected goals", "progressive pass", "shot map", "ball recovery",
        "second ball", "rest defense", "wing overload", "transition phase",
    ]
    n = len(phrases)
    rng = np.random.default_rng(random_seed)

    # 1. Draw unigram frequencies between 20 and 100
    freqs = rng.integers(20, 101, size=n)

    # 2. Draw symmetric co-occurrence counts
    cooc = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            max_co = int(min(freqs[i], freqs[j]))
            if max_co <= 0:
                c = 0
            else:
                c = int(rng.integers(0, max_co + 1))
            cooc[i, j] = cooc[j, i] = c

    return phrases, freqs, cooc


def compute_ppmi(freqs, cooc):
    # Estimate probabilities from the co-occurrence counts
    cooc = np.asarray(cooc, dtype=float)
    total = cooc.sum()
    # Avoid division by zero
    if total == 0:
        raise ValueError("Total co-occurrence count is zero.")

    p_ij = cooc / total                 # joint probabilities
    p_i = p_ij.sum(axis=1)              # marginal probabilities

    n = cooc.shape[0]
    ppmi = np.zeros_like(p_ij)

    for i in range(n):
        for j in range(n):
            if cooc[i, j] <= 0:
                ppmi[i, j] = 0.0
                continue
            denom = p_i[i] * p_i[j]
            if denom <= 0:
                ppmi[i, j] = 0.0
                continue
            pmi = math.log2(p_ij[i, j] / denom)
            ppmi[i, j] = max(pmi, 0.0)

    return ppmi


def main():
    # A. build synthetic data
    phrases, freqs, cooc = build_counts()
    df_freq = pd.DataFrame({"phrase": phrases, "freq": freqs})
    df_cooc = pd.DataFrame(cooc, index=phrases, columns=phrases)

    print("===== PHRASES AND FREQUENCIES =====")
    print(df_freq.to_string(index=False))

    print("\n===== CO-OCCURRENCE MATRIX (COUNTS) =====")
    print(df_cooc.to_string())

    # B. compute PPMI
    ppmi = compute_ppmi(freqs, cooc)
    df_ppmi = pd.DataFrame(ppmi, index=phrases, columns=phrases)

    print("\n===== PPMI MATRIX =====")
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(df_ppmi.to_string())

    # C. visualization: PPMI heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ppmi, aspect='auto')
    ax.set_xticks(range(len(phrases)))
    ax.set_yticks(range(len(phrases)))
    ax.set_xticklabels(phrases, rotation=90, fontsize=6)
    ax.set_yticklabels(phrases, fontsize=6)
    ax.set_title("PPMI Heatmap for Football Phrases")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
