#!/usr/bin/env python3
"""
Generate KVTuner per-layer quantization config files for thesis experiments.

Generates the --kvtuner-layer-bits argument value (JSON int list) for each
quantization strategy defined in thesis_test_plan.md section 3.3.

Usage:
    # Generate all configs for Qwen2.5-7B (28 layers)
    python3 scripts/pd_disagg_test/generate_quant_configs.py --num-layers 28

    # Generate for a specific strategy
    python3 scripts/pd_disagg_test/generate_quant_configs.py --num-layers 28 --strategy mixed-B

    # Custom output directory
    python3 scripts/pd_disagg_test/generate_quant_configs.py --num-layers 28 --output-dir ./my_configs

Output files:
    uniform-8bit.json       All layers 8-bit
    uniform-4bit.json       All layers 4-bit
    mixed-A.json            First/last 5 layers 8-bit + middle 4-bit
    mixed-B.json            Sensitivity top-5 layers 8-bit + rest 4-bit
    mixed-C.json            Sensitivity top-10 layers 8-bit + rest 4-bit
    mixed-D.json            Top-5 layers 8-bit + middle 4-bit + bottom 2-bit
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Strategy definitions (from thesis_test_plan.md section 3.3)
# ---------------------------------------------------------------------------

STRATEGIES = {
    "uniform-8bit": {
        "description": "All layers 8-bit",
        "avg_bits": 8.0,
        "compression": "~1.94x",
    },
    "uniform-4bit": {
        "description": "All layers 4-bit",
        "avg_bits": 4.0,
        "compression": "~3.77x",
    },
    "mixed-A": {
        "description": "First/last 5 layers 8-bit + middle 4-bit",
        "avg_bits": "5.14",
        "compression": "~3.0x",
    },
    "mixed-B": {
        "description": "Sensitivity top-5 layers 8-bit + rest 4-bit",
        "avg_bits": "~5.14",
        "compression": "~3.0x",
    },
    "mixed-C": {
        "description": "Sensitivity top-10 layers 8-bit + rest 4-bit",
        "avg_bits": "~6.0",
        "compression": "~2.5x",
    },
    "mixed-D": {
        "description": "Top-5 layers 8-bit + middle 4-bit + bottom 2-bit",
        "avg_bits": "~4.0",
        "compression": "~3.5x",
    },
}

# Sensitivity ranking for Qwen2.5-7B (28 layers).
# Layers closer to input/output tend to be more sensitive.
# This is an empirical heuristic; for a real system, run kvtuner_offline_calib.py.
# Format: list of layer indices ordered from most sensitive to least.
SENSITIVITY_RANKING_QWEN_7B = [
    0, 1, 2, 27, 26, 3, 25, 24, 4, 5,
    23, 22, 21, 6, 7, 20, 19, 8, 18, 17,
    9, 10, 16, 15, 11, 12, 13, 14,
]


def generate_uniform(num_layers: int, nbits: int) -> list:
    return [nbits] * num_layers


def generate_mixed_a(num_layers: int) -> list:
    """First 5 + last 5 layers 8-bit, middle layers 4-bit."""
    config = [4] * num_layers
    for i in range(min(5, num_layers)):
        config[i] = 8
    for i in range(max(0, num_layers - 5), num_layers):
        config[i] = 8
    return config


def generate_mixed_b(num_layers: int, ranking: list) -> list:
    """Sensitivity top-5 layers 8-bit, rest 4-bit."""
    config = [4] * num_layers
    top_n = min(5, num_layers)
    for i in range(top_n):
        config[ranking[i]] = 8
    return config


def generate_mixed_c(num_layers: int, ranking: list) -> list:
    """Sensitivity top-10 layers 8-bit, rest 4-bit."""
    config = [4] * num_layers
    top_n = min(10, num_layers)
    for i in range(top_n):
        config[ranking[i]] = 8
    return config


def generate_mixed_d(num_layers: int, ranking: list) -> list:
    """Top-5 layers 8-bit, middle layers 4-bit, bottom layers 2-bit."""
    config = [4] * num_layers
    top_n = min(5, num_layers)
    # Top-5 most sensitive → 8-bit
    for i in range(top_n):
        config[ranking[i]] = 8
    # Bottom 10 least sensitive → 2-bit
    bottom_n = min(10, num_layers - top_n)
    for i in range(bottom_n):
        config[ranking[-(i + 1)]] = 2
    return config


def build_sensitivity_ranking(num_layers: int) -> list:
    """Build a sensitivity ranking for arbitrary model sizes.

    Uses the heuristic: layers near input (0,1,2,...) and output (N-1,N-2,...)
    are most sensitive, middle layers are least sensitive.
    """
    left = list(range(num_layers))
    right = []
    remaining = list(range(num_layers))
    # Interleave from both ends: 0, N-1, 1, N-2, 2, N-3, ...
    ranking = []
    i = 0
    j = num_layers - 1
    toggle = True
    while i <= j:
        if toggle:
            ranking.append(i)
            i += 1
        else:
            ranking.append(j)
            j -= 1
        toggle = not toggle
    return ranking


def generate_strategy(name: str, num_layers: int, ranking: list) -> list:
    """Generate a per-layer bit config for the given strategy."""
    if name == "uniform-8bit":
        return generate_uniform(num_layers, 8)
    elif name == "uniform-4bit":
        return generate_uniform(num_layers, 4)
    elif name == "mixed-A":
        return generate_mixed_a(num_layers)
    elif name == "mixed-B":
        return generate_mixed_b(num_layers, ranking)
    elif name == "mixed-C":
        return generate_mixed_c(num_layers, ranking)
    elif name == "mixed-D":
        return generate_mixed_d(num_layers, ranking)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def print_config(bits: list, name: str, num_layers: int) -> None:
    """Print a summary of the config."""
    from collections import Counter
    dist = Counter(bits)
    avg = sum(bits) / len(bits)

    print(f"  {name}: avg={avg:.2f}-bit  ", end="")
    for nbits in sorted(dist.keys()):
        count = dist[nbits]
        pct = count / num_layers * 100
        print(f"{nbits}-bit:{count}L({pct:.0f}%)  ", end="")
    print()
    print(f"    bits={json.dumps(bits)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate KVTuner per-layer quantization configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of transformer layers (default: 28 for Qwen2.5-7B)")
    parser.add_argument("--strategy",
                        help="Generate only this strategy (default: all)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for JSON files (default: same dir as this script)")
    parser.add_argument("--shell-export", action="store_true",
                        help="Also print shell export commands for KVTUNER_LAYER_BITS")
    args = parser.parse_args()

    # Sensitivity ranking
    if args.num_layers == 28:
        ranking = SENSITIVITY_RANKING_QWEN_7B
    else:
        ranking = build_sensitivity_ranking(args.num_layers)

    # Determine strategies to generate
    if args.strategy:
        strategies = {args.strategy: STRATEGIES[args.strategy]}
    else:
        strategies = STRATEGIES

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "quant")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating quantization configs for {args.num_layers} layers")
    print(f"Output directory: {output_dir}")
    print(f"Sensitivity ranking: {ranking}")
    print()

    generated = {}

    for name, info in strategies.items():
        bits = generate_strategy(name, args.num_layers, ranking)
        generated[name] = bits

        # Print summary
        print_config(bits, name, args.num_layers)

        # Write JSON file (just the int array, single-line for shell compatibility)
        json_path = os.path.join(output_dir, f"{name}.json")
        with open(json_path, "w") as f:
            json.dump(bits, f, separators=(",", ": "))

        # Also write a detailed config with metadata
        detailed = {
            "strategy": name,
            "description": info["description"],
            "num_layers": args.num_layers,
            "avg_bits": round(sum(bits) / len(bits), 2),
            "bits": bits,
            "sensitivity_ranking": ranking,
        }
        detailed_path = os.path.join(output_dir, f"{name}.detail.json")
        with open(detailed_path, "w") as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False)

        if args.shell_export:
            bits_str = json.dumps(bits)
            print(f"    export KVTUNER_LAYER_BITS='{bits_str}'")
        print()

    print(f"All configs written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
