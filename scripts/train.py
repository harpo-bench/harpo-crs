#!/usr/bin/env python3
"""
HARPO: Main Training Script

Optimizing Conversational Recommendation for User-Aligned Quality
via Hierarchical Preference Learning

Usage:
    # Single GPU
    python scripts/train.py --sft data/redial/sft_data.json --pref data/redial/preference_data.json
    
    # Multi-GPU with Accelerate (recommended)
    accelerate launch --config_file configs/accelerate_config.yaml \
        scripts/train.py --sft data/redial/sft_data.json --pref data/redial/preference_data.json
    
    # Resume from checkpoint
    python scripts/train.py --sft data/redial/sft_data.json --pref data/redial/preference_data.json \
        --resume sft_final --skip-stages sft
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def main():
    parser = argparse.ArgumentParser(
        description="HARPO Training - Conversational Recommendation with Hierarchical Preference Learning"
    )
    parser.add_argument("--sft", required=True, help="SFT data path (JSON)")
    parser.add_argument("--pref", required=True, help="Preference data path (JSON)")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                        help="Base model name or path")
    parser.add_argument("--skip-stages", nargs="*", default=[], 
                        choices=["sft", "charm", "star", "maven"],
                        help="Stages to skip")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint name (e.g., 'sft_final')")
    parser.add_argument("--test-split", type=float, default=0.1, 
                        help="Test split ratio")
    parser.add_argument("--validate-only", action="store_true", 
                        help="Only validate data format")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HARPO: Hierarchical Preference Learning for Conversational Recommendation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    
    # Import torch after setting environment
    import torch
    
    # Disable dynamo
    torch._dynamo.config.suppress_errors = True
    try:
        torch._dynamo.disable()
    except:
        pass
    
    # Check GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU detected!")
    
    # Validate data
    print("\n[Step 1] Validating data...")
    
    if not os.path.exists(args.sft):
        print(f"ERROR: SFT data not found: {args.sft}")
        sys.exit(1)
    if not os.path.exists(args.pref):
        print(f"ERROR: Preference data not found: {args.pref}")
        sys.exit(1)
    
    with open(args.sft) as f:
        sft_data = json.load(f)
    with open(args.pref) as f:
        pref_data = json.load(f)
    
    print(f"  ✓ SFT data: {len(sft_data)} examples")
    print(f"  ✓ Preference data: {len(pref_data)} pairs")
    
    # Validate format
    required_sft = ["input", "output", "vtos"]
    required_pref = ["context", "chosen", "rejected", "hierarchical_rewards"]
    
    if not all(k in sft_data[0] for k in required_sft):
        print(f"ERROR: SFT data missing fields. Required: {required_sft}")
        sys.exit(1)
    if pref_data and not all(k in pref_data[0] for k in required_pref):
        print(f"ERROR: Preference data missing fields. Required: {required_pref}")
        sys.exit(1)
    
    print("  ✓ Data format validated")
    
    if args.validate_only:
        print("\nValidation complete!")
        sys.exit(0)
    
    # Prepare data splits
    print("\n[Step 2] Preparing data splits...")
    os.makedirs(args.output, exist_ok=True)
    
    n_sft = len(sft_data)
    n_pref = len(pref_data)
    
    test_size_sft = int(n_sft * args.test_split)
    test_size_pref = int(n_pref * args.test_split)
    
    train_sft = sft_data[:-test_size_sft] if test_size_sft > 0 else sft_data
    test_sft = sft_data[-test_size_sft:] if test_size_sft > 0 else []
    
    train_pref = pref_data[:-test_size_pref] if test_size_pref > 0 else pref_data
    test_pref = pref_data[-test_size_pref:] if test_size_pref > 0 else []
    
    print(f"  SFT: {len(train_sft)} train, {len(test_sft)} test")
    print(f"  Preference: {len(train_pref)} train, {len(test_pref)} test")
    
    # Save splits
    train_sft_path = os.path.join(args.output, "train_sft.json")
    test_sft_path = os.path.join(args.output, "test_sft.json")
    train_pref_path = os.path.join(args.output, "train_pref.json")
    test_pref_path = os.path.join(args.output, "test_pref.json")
    
    with open(train_sft_path, "w") as f:
        json.dump(train_sft, f)
    with open(test_sft_path, "w") as f:
        json.dump(test_sft, f)
    with open(train_pref_path, "w") as f:
        json.dump(train_pref, f)
    with open(test_pref_path, "w") as f:
        json.dump(test_pref, f)
    
    # Handle resume
    print("\n[Step 3] Starting training...")
    
    from harpo.training import run_full_training
    
    skip_stages = list(args.skip_stages) if args.skip_stages else []
    resume_checkpoint = None
    model_path = args.model
    
    if args.resume:
        resume_checkpoint = os.path.join(args.output, "checkpoints", args.resume)
        if os.path.exists(resume_checkpoint):
            print(f"✓ Resuming from checkpoint: {resume_checkpoint}")
            
            # Use checkpoint's base_model for offline loading
            checkpoint_model_path = os.path.join(resume_checkpoint, "base_model")
            if os.path.exists(checkpoint_model_path):
                model_path = checkpoint_model_path
                print(f"  → Using model from checkpoint")
            
            # Auto-skip completed stages
            if "sft" in args.resume and "sft" not in skip_stages:
                skip_stages.append("sft")
            if "charm" in args.resume:
                for s in ["sft", "charm"]:
                    if s not in skip_stages:
                        skip_stages.append(s)
            if "star" in args.resume:
                for s in ["sft", "charm", "star"]:
                    if s not in skip_stages:
                        skip_stages.append(s)
        else:
            print(f"⚠ Checkpoint not found: {resume_checkpoint}")
            resume_checkpoint = None
    
    # Try GPU config
    gpu_config = None
    try:
        from harpo.gpu_config import detect_gpu_config
        gpu_config = detect_gpu_config()
    except ImportError:
        pass
    
    # Run training
    model, trainer = run_full_training(
        sft_data_path=train_sft_path,
        preference_data_path=train_pref_path,
        model_name=model_path,
        output_dir=args.output,
        skip_stages=skip_stages,
        gpu_config=gpu_config,
        use_accelerate=True,
        resume_checkpoint=resume_checkpoint
    )
    
    # Evaluation
    print("\n[Step 4] Final Evaluation...")
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    from harpo.evaluation import HARPOMTv2Evaluator
    
    eval_cache_dir = os.path.join(args.output, "eval_cache")
    os.makedirs(eval_cache_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = HARPOMTv2Evaluator(actual_model, device)
    
    results = evaluator.run_full_evaluation(
        test_sft, test_pref,
        max_samples=100,
        cache_dir=eval_cache_dir
    )
    
    # Save results
    results_path = os.path.join(args.output, "evaluation_results.json")
    
    def serialize_for_json(obj):
        if hasattr(obj, 'value'):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_for_json(item) for item in obj]
        else:
            return obj
    
    from dataclasses import asdict
    with open(results_path, "w") as f:
        json.dump(serialize_for_json(asdict(results)), f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Output: {args.output}")
    print(f"Checkpoints: {args.output}/checkpoints/")
    print(f"Results: {results_path}")
    print(f"\nKey Metrics:")
    print(f"  Recall@10: {results.recall_at_10:.4f}")
    print(f"  MRR: {results.mrr:.4f}")
    print(f"  User Satisfaction: {results.user_satisfaction:.4f}")
    print(f"\nTo chat:")
    print(f"  python scripts/chat.py --checkpoint {args.output}/checkpoints/final")


if __name__ == "__main__":
    main()