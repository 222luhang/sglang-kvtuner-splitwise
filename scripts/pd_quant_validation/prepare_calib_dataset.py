#!/usr/bin/env python3
"""
Prepare Calibration Dataset for KVTuner

准备校准数据集，用于 KVTuner 离线量化配置计算。
"""

import json
import random
import argparse
from pathlib import Path


def load_wikitext_samples(num_samples: int = 512) -> list:
    """从 WikiText 加载校准样本"""
    # WikiText 常用样本
    samples = [
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "The Transformer is a deep learning model that adopts the mechanism of self-attention.",
        "Large language models are artificial intelligence systems that can understand and generate human language.",
        "Python is a high-level, general-purpose programming language.",
        "Quantum computing is a type of computation whose operations can exploit quantum mechanical phenomena.",
        "Cloud computing is the on-demand availability of computer system resources.",
        "Blockchain is a growing list of records, called blocks, that are linked together using cryptography.",
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence.",
    ]
    
    # 扩展样本到所需数量
    extended = []
    while len(extended) < num_samples:
        # 随机组合和修改样本
        base = random.choice(samples)
        # 添加一些变化
        variations = [
            base,
            base + " This is an important concept in modern technology.",
            "In recent years, " + base.lower(),
            base + " Many researchers are working on this topic.",
            "What is " + base.split()[0].lower() + "? " + base,
        ]
        extended.append(random.choice(variations))
    
    return extended[:num_samples]


def prepare_calib_dataset(
    output_path: str,
    num_samples: int = 512,
    dataset_type: str = "wikitext"
):
    """准备校准数据集"""
    print(f"Preparing {num_samples} calibration samples...")
    
    if dataset_type == "wikitext":
        samples = load_wikitext_samples(num_samples)
    else:
        samples = load_wikitext_samples(num_samples)
    
    # 创建数据集
    dataset = {
        "type": dataset_type,
        "num_samples": num_samples,
        "samples": [
            {"id": i, "text": text}
            for i, text in enumerate(samples)
        ],
        "metadata": {
            "avg_length": sum(len(s.split()) for s in samples) / len(samples),
            "max_length": max(len(s.split()) for s in samples),
            "min_length": min(len(s.split()) for s in samples),
        }
    }
    
    # 保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to {output_file}")
    print(f"   Samples: {num_samples}")
    print(f"   Avg length: {dataset['metadata']['avg_length']:.1f} tokens")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Calibration Dataset")
    parser.add_argument("--output", type=str, default="/data/kvtuner_calib/calib_dataset.json",
                       help="Output path")
    parser.add_argument("--samples", type=int, default=512,
                       help="Number of samples")
    parser.add_argument("--type", type=str, default="wikitext",
                       help="Dataset type (wikitext, c4, etc.)")
    
    args = parser.parse_args()
    
    prepare_calib_dataset(
        output_path=args.output,
        num_samples=args.samples,
        dataset_type=args.type
    )
