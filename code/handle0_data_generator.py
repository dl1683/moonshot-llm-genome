"""
HANDLE-0 Affine Ledger: trajectory data generator.

Generates training/evaluation datasets for the Affine Ledger task.
Each trajectory is a sequence of text chunks with KV-cache resets between them.
Produces counterfactual groups (identical final chunks, different histories,
different correct answers) with balanced renderer crossing.

Depends on handle0_affine_ledger.py for the reference interpreter.
"""

import itertools
import json
import random
import sys
from pathlib import Path

from handle0_affine_ledger import (
    P, UNITS, AFFINES, apply_affine, compose_affine, eval_query, QUERIES,
)

OUTPUT_TOKENS = ["A", "B", "C", "D", "E"]

QUERY_TEMPLATES = {
    ("READ", "V0"): "READ V0",
    ("READ", "V1"): "READ V1",
    ("EVAL", "M0", 0): "EVAL M0 AT 0",
    ("EVAL", "M0", 1): "EVAL M0 AT 1",
    ("EVAL", "M1", 0): "EVAL M1 AT 0",
    ("EVAL", "M1", 1): "EVAL M1 AT 1",
    ("APPLY", "M0", "V0"): "APPLY M0 TO V0",
    ("APPLY", "M0", "V1"): "APPLY M0 TO V1",
    ("APPLY", "M1", "V0"): "APPLY M1 TO V0",
    ("APPLY", "M1", "V1"): "APPLY M1 TO V1",
    ("APPLY_COMPOSED", "M0", "M1", "V0"): "APPLY M0 AFTER M1 TO V0",
    ("APPLY_COMPOSED", "M0", "M1", "V1"): "APPLY M0 AFTER M1 TO V1",
    ("APPLY_COMPOSED", "M1", "M0", "V0"): "APPLY M1 AFTER M0 TO V0",
    ("APPLY_COMPOSED", "M1", "M0", "V1"): "APPLY M1 AFTER M0 TO V1",
}


def random_state(rng):
    """Sample a uniformly random machine state."""
    return (
        rng.randrange(P),
        rng.randrange(P),
        AFFINES[rng.randrange(len(AFFINES))],
        AFFINES[rng.randrange(len(AFFINES))],
    )


def generate_init_chunk(v0, v1, m0, m1):
    """Generate initialization commands that set all four registers."""
    lines = [
        f"SETV V0 {v0}",
        f"SETV V1 {v1}",
        f"SETM M0 {m0[0]} {m0[1]}",
        f"SETM M1 {m1[0]} {m1[1]}",
    ]
    return lines


def generate_update_sequence(rng, n_updates, require_compose=True, require_transport=True):
    """Generate a random sequence of update commands.
    Guarantees at least one COMPOSE and one TRANSPORT if required."""
    ops = []
    has_compose = False
    has_transport = False

    for i in range(n_updates):
        if i == n_updates - 2 and require_transport and not has_transport:
            op_type = "TRANSPORT"
        elif i == n_updates - 1 and require_compose and not has_compose:
            op_type = "COMPOSE"
        else:
            op_type = rng.choice(["SETV", "SETM", "TRANSPORT", "COMPOSE", "SWAPV", "SWAPM"])

        if op_type == "SETV":
            reg = rng.choice(["V0", "V1"])
            val = rng.randrange(P)
            ops.append(f"SETV {reg} {val}")
        elif op_type == "SETM":
            reg = rng.choice(["M0", "M1"])
            a = rng.choice(UNITS)
            b = rng.randrange(P)
            ops.append(f"SETM {reg} {a} {b}")
        elif op_type == "TRANSPORT":
            vi = rng.choice(["V0", "V1"])
            mj = rng.choice(["M0", "M1"])
            ops.append(f"TRANSPORT {vi} THROUGH {mj}")
            has_transport = True
        elif op_type == "COMPOSE":
            mi = rng.choice(["M0", "M1"])
            mj = rng.choice(["M0", "M1"])
            if mi == mj:
                mj = "M1" if mi == "M0" else "M0"
            ops.append(f"COMPOSE {mi} AFTER {mj}")
            has_compose = True
        elif op_type == "SWAPV":
            ops.append("SWAPV")
        elif op_type == "SWAPM":
            ops.append("SWAPM")

    return ops


def execute_commands(init_state, commands):
    """Execute a command sequence on a state, return final state."""
    v0, v1, m0, m1 = init_state

    for cmd in commands:
        parts = cmd.split()
        op = parts[0]

        if op == "SETV":
            reg, val = parts[1], int(parts[2])
            if reg == "V0":
                v0 = val
            else:
                v1 = val
        elif op == "SETM":
            reg, a, b = parts[1], int(parts[2]), int(parts[3])
            m = (a, b)
            if reg == "M0":
                m0 = m
            else:
                m1 = m
        elif op == "TRANSPORT":
            vi, mj = parts[1], parts[3]
            v = v0 if vi == "V0" else v1
            m = m0 if mj == "M0" else m1
            result = apply_affine(m, v)
            if vi == "V0":
                v0 = result
            else:
                v1 = result
        elif op == "COMPOSE":
            mi, mj = parts[1], parts[3]
            ma = m0 if mi == "M0" else m1
            mb = m0 if mj == "M0" else m1
            composed = compose_affine(ma, mb)
            if mi == "M0":
                m0 = composed
            else:
                m1 = composed
        elif op == "SWAPV":
            v0, v1 = v1, v0
        elif op == "SWAPM":
            m0, m1 = m1, m0

    return (v0, v1, m0, m1)


def make_render_line(perm):
    """Create RENDER line from a permutation mapping logical -> token index."""
    parts = []
    for logical in range(P):
        token = OUTPUT_TOKENS[perm[logical]]
        parts.append(f"{logical}={token}")
    return "RENDER " + " ".join(parts)


def make_final_chunk(query, perm):
    """Create the final query chunk with renderer."""
    query_text = QUERY_TEMPLATES[query]
    render_line = make_render_line(perm)
    return [render_line, f"QUERY {query_text}", "ANSWER"]


def generate_trajectory(rng, n_updates=6):
    """Generate one complete trajectory with random state, commands, query, and renderer."""
    init = random_state(rng)
    commands = generate_update_sequence(rng, n_updates)
    final_state = execute_commands(init, commands)

    query = QUERIES[rng.randrange(len(QUERIES))]
    logical_answer = eval_query(final_state, query)

    perm = list(range(P))
    rng.shuffle(perm)
    output_token = OUTPUT_TOKENS[perm[logical_answer]]

    init_cmds = generate_init_chunk(*init)
    final_chunk = make_final_chunk(query, perm)

    chunks = []
    for cmd in init_cmds:
        chunks.append({"text": cmd, "expected_output": "ACK"})
    for cmd in commands:
        chunks.append({"text": cmd, "expected_output": "ACK"})
    chunks.append({"text": "\n".join(final_chunk), "expected_output": output_token})

    return {
        "init_state": {"v0": init[0], "v1": init[1],
                       "m0": list(init[2]), "m1": list(init[3])},
        "commands": init_cmds + commands,
        "final_state": {"v0": final_state[0], "v1": final_state[1],
                        "m0": list(final_state[2]), "m1": list(final_state[3])},
        "query": QUERY_TEMPLATES[query],
        "renderer": perm,
        "logical_answer": logical_answer,
        "output_token": output_token,
        "chunks": chunks,
    }


def generate_counterfactual_group(rng, n_updates=6):
    """Generate a group of 5 trajectories with identical final chunks but
    different histories producing all 5 logical answers."""
    commands_template = generate_update_sequence(rng, n_updates)
    query = QUERIES[rng.randrange(len(QUERIES))]
    perm = list(range(P))
    rng.shuffle(perm)

    group = []
    attempts = 0
    seen_answers = set()

    while len(seen_answers) < P and attempts < 5000:
        attempts += 1
        init = random_state(rng)
        final_state = execute_commands(init, commands_template)
        logical_answer = eval_query(final_state, query)

        if logical_answer not in seen_answers:
            seen_answers.add(logical_answer)
            output_token = OUTPUT_TOKENS[perm[logical_answer]]
            init_cmds = generate_init_chunk(*init)
            final_chunk = make_final_chunk(query, perm)

            chunks = []
            for cmd in init_cmds:
                chunks.append({"text": cmd, "expected_output": "ACK"})
            for cmd in commands_template:
                chunks.append({"text": cmd, "expected_output": "ACK"})
            chunks.append({"text": "\n".join(final_chunk), "expected_output": output_token})

            group.append({
                "init_state": {"v0": init[0], "v1": init[1],
                               "m0": list(init[2]), "m1": list(init[3])},
                "commands": init_cmds + commands_template,
                "final_state": {"v0": final_state[0], "v1": final_state[1],
                                "m0": list(final_state[2]), "m1": list(final_state[3])},
                "query": QUERY_TEMPLATES[query],
                "renderer": perm,
                "logical_answer": logical_answer,
                "output_token": output_token,
                "chunks": chunks,
            })

    if len(seen_answers) < P:
        return None
    return group


def generate_dataset(n_trajectories=1000, n_cf_groups=100, seed=42):
    """Generate a full dataset with individual trajectories and counterfactual groups."""
    rng = random.Random(seed)

    trajectories = []
    for _ in range(n_trajectories):
        n_updates = rng.randint(4, 12)
        traj = generate_trajectory(rng, n_updates)
        trajectories.append(traj)

    cf_groups = []
    for _ in range(n_cf_groups):
        n_updates = rng.randint(4, 8)
        group = generate_counterfactual_group(rng, n_updates)
        if group is not None:
            cf_groups.append(group)

    return {"trajectories": trajectories, "counterfactual_groups": cf_groups}


def verify_dataset(dataset):
    """Run basic integrity checks on a generated dataset."""
    print(f"Trajectories: {len(dataset['trajectories'])}")
    print(f"Counterfactual groups: {len(dataset['counterfactual_groups'])}")

    answer_dist = [0] * P
    for traj in dataset["trajectories"]:
        answer_dist[traj["logical_answer"]] += 1
    print(f"Answer distribution: {answer_dist}")

    renderer_counts = [[0] * P for _ in range(P)]
    for traj in dataset["trajectories"]:
        la = traj["logical_answer"]
        ti = OUTPUT_TOKENS.index(traj["output_token"])
        renderer_counts[la][ti] += 1
    print(f"Renderer balance (logical x token):")
    for row in renderer_counts:
        print(f"  {row}")

    cf_complete = sum(1 for g in dataset["counterfactual_groups"] if len(g) == P)
    print(f"Complete counterfactual groups (all 5 answers): {cf_complete}/{len(dataset['counterfactual_groups'])}")

    for gi, group in enumerate(dataset["counterfactual_groups"]):
        if len(group) < 2:
            continue
        final_a = group[0]["chunks"][-1]["text"]
        for traj in group[1:]:
            final_b = traj["chunks"][-1]["text"]
            if final_a != final_b:
                print(f"  WARNING: CF group {gi} has mismatched final chunks!")
                break

    return True


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"=== HANDLE-0 Affine Ledger: Data Generator (seed={seed}) ===\n")

    dataset = generate_dataset(seed=seed)
    verify_dataset(dataset)

    out_path = out_dir / f"handle0_dataset_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=1)
    print(f"\nDataset written to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
