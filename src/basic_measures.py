import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.configuration import MADRID_FILE, SEGOVIA_FILE

madrid  = pd.read_excel(MADRID_FILE)
segovia = pd.read_excel(SEGOVIA_FILE)
madrid  = madrid[madrid['Relationship to IE'] != 'Professor/Lecturer'].copy()
segovia = segovia[segovia['Relationship to IE'] != 'Professor/Lecturer'].copy()
madrid['is_BAM']  = madrid['What degree are you studying?'].str.contains('BAM', na=False)

seg_bam = segovia.copy()
mad_bam = madrid[madrid['is_BAM']].copy()

# Using our 7-question binary scoring (same as q1_learning.py)
# Pre correct answers
from src.configuration import SEGOVIA_FILE, MADRID_FILE

PRE_DEFS = [
    ("Q1", "Which statement is correct?",
     "A qubit can be in a combination \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),
    ("Q2", "A qubit is prepared so that when measuring it, it gives outcome |0\u27e9 with probability 0.25. "
            "What is the probability of obtaining outcome |1\u27e9  (considering the same state preparation)? "
            "Assume only outcomes of states |0\u27e9 or |1\u27e9.", 0.75),
    ("Q3", "A qubit starts in state \u22230\u27e9. You apply a Hadamard (H) gate to obtain a superposition state. "
            "Which statement best describes the result?",
            "Measuring the qubit can yield |0\u27e9 or |1\u27e9 with equal probabilities"),
    ("Q4", "Which statement best captures entanglement of two qubits?",
            "Two entangled qubits have correlations that cannot be described by a product state (not separable)"),
    ("Q5", "Starting from |00\u27e9, what quantum logic gates are required (at least) to create "
            "an entangled state of two qubits?", "A Hadamard gate and a CNOT gate"),
    ("Q6", "How does the act of measuring affect a quantum system that is in superposition? "
            "Assume the computational basis.",
            "Measurement \u201ccollapses\u201d the quantum system into an (eigen)state of the measured observable"),
    ("Q7", "Quantum computers often require strong shielding and (for many platforms) cryogenic "
            "temperatures mainly to  ",
            "Reduce thermal noise and environmental interactions, which minimizes error rates and decoherence"),
]
POST_DEFS = [
    ("Q1", "Which statement is correct? 2",
     "A qubit state can be written as \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),
    ("Q2", "A qubit is in the state \u2223\u03c8\u27e9= \u221a3/2 |0\u27e9  + 1/2 |1\u27e9. If you measure this qubit, "
            "what is the probability of obtaining the state |1\u27e9?  Assume only outcomes for states |0\u27e9 or |1\u27e9.", "1/4"),
    ("Q3", " Which description best matches the meaning of a qubit in superposition "
            "(\u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1)? Assume the computational basis.",
            "After measurement, the qubit will yield |0\u27e9 or |1\u27e9 with probabilities |\u03b1|\u00b2 and |\u03b2|\u00b2, respectively"),
    ("Q4", "Which statement best captures entanglement of two qubits? 2",
            "Two entangled qubits have correlations that cannot be described by a product state "
            "(i.e., the joint state is not separable)"),
    ("Q5", "Starting from a qubit in state |0\u27e9, which quantum gate operation can prepare "
            "a superposition state such as (|0\u27e9+|1\u27e9)/\u221a2?", "Apply a Hadamard gate to the qubit"),
    ("Q6", "Which statement best describes the effect of measurement on a quantum state? "
            "Assume the computational basis.",
            "After the measurement, the quantum system is one of its (eigen)states of the measured observable"),
    ("Q7", "Maintaining quantum coherence in a quantum computer often requires:",
            "Isolation from radiation and vibrations; and (often) very low temperatures"),
]

for df in [seg_bam, mad_bam]:
    pre_cols, post_cols = [], []
    for qk, col, ans in PRE_DEFS:
        df[f'pre_{qk}']  = (df[col] == ans).astype(float)
        df.loc[df[col].isna(), f'pre_{qk}'] = np.nan
        pre_cols.append(f'pre_{qk}')
    for qk, col, ans in POST_DEFS:
        df[f'post_{qk}'] = (df[col] == ans).astype(float)
        df.loc[df[col].isna(), f'post_{qk}'] = np.nan
        post_cols.append(f'post_{qk}')
    df['pre_total']  = df[pre_cols].sum(axis=1)
    df['post_total'] = df[post_cols].sum(axis=1)

# Filter both pre and post answered
seg_f = seg_bam.dropna(subset=pre_cols + post_cols).copy()
mad_f = mad_bam.dropna(subset=pre_cols + post_cols).copy()

print("TEST SCORE DESCRIPTIVES (doc-verified, out of 7, both tests answered)")
print(f"{'Group':<25} {'n':>4}  {'Pre mean':>9}  {'Pre SD':>7}  {'Post mean':>10}  {'Post SD':>8}")
print("-" * 70)
for name, df in [('Segovia BAM (control)', seg_f), ('Madrid BAM (treatment)', mad_f)]:
    print(f"{name:<25} {len(df):>4}  {df['pre_total'].mean():>9.2f}  {df['pre_total'].std():>7.2f}"
          f"  {df['post_total'].mean():>10.2f}  {df['post_total'].std():>8.2f}")
