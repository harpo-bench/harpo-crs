#!/usr/bin/env python3
"""
HARPO: Standalone Evaluation Script

Run comprehensive evaluation on a trained checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final
    
    python scripts/evaluate.py \
        --checkpoint outputs/checkpoints/final \
        --test-data data/redial/test_sft.json \
        --max-samples 200
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch


def main():
    parser = argparse.ArgumentParser(description="HARPO Evaluation")
    parser.add_argument("--checkpoint", required=True, 
                        help="Path to checkpoint directory")
    parser.add_argument("--test-data", default=None, 
                        help="Test data path (JSON)")
    parser.add_argument("--pref-data", default=None,
                        help="Preference test data (optional)")
    parser.add_argument("--output", default=None, 
                        help="Output directory (default: checkpoint/evaluation)")
    parser.add_argument("--max-samples", type=int, default=100, 
                        help="Max samples per evaluation")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear evaluation cache")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HARPO: Comprehensive Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    base_model_path = os.path.join(args.checkpoint, "base_model")
    components_path = os.path.join(args.checkpoint, "components.pt")
    
    if not os.path.exists(base_model_path):
        print(f"ERROR: base_model not found: {base_model_path}")
        sys.exit(1)
    
    print(f"✓ Found base_model")
    if os.path.exists(components_path):
        print(f"✓ Found components.pt")
    
    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ Using CPU")
    
    # Set output
    if args.output is None:
        args.output = os.path.join(args.checkpoint, "evaluation")
    os.makedirs(args.output, exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_dir = os.path.join(args.output, "eval_cache")
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✓ Cleared cache: {cache_dir}")
    
    # Load test data
    print("\n[Step 1] Loading test data...")
    
    test_sft = []
    test_pref = []
    
    if args.test_data and os.path.exists(args.test_data):
        with open(args.test_data) as f:
            test_sft = json.load(f)
        print(f"  Loaded {len(test_sft)} test examples")
    else:
        # Try to find test data in checkpoint parent
        parent_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        test_path = os.path.join(parent_dir, "test_sft.json")
        if os.path.exists(test_path):
            with open(test_path) as f:
                test_sft = json.load(f)
            print(f"  Found test data: {test_path}")
    
    if args.pref_data and os.path.exists(args.pref_data):
        with open(args.pref_data) as f:
            test_pref = json.load(f)
    
    if not test_sft:
        print("WARNING: No test data found. Using empty dataset.")
    
    # Load model
    print("\n[Step 2] Loading model...")
    
    from harpo.config import ModelConfig, TrainingConfig
    from harpo.model import HARPOMTv2
    
    model_config = ModelConfig(model_name=base_model_path)
    training_config = TrainingConfig()
    
    model = HARPOMTv2(model_config, training_config)
    model.load_base_model(device)
    
    # Load components
    if os.path.exists(components_path):
        checkpoint = torch.load(components_path, map_location=device)
        
        components = [
            ("bridge", "bridge_state_dict"),
            ("star", "star_state_dict"),
            ("charm", "charm_state_dict"),
            ("maven", "maven_state_dict"),
            ("vto_head", "vto_head_state_dict"),
        ]
        
        for comp_name, state_key in components:
            if state_key in checkpoint:
                comp = getattr(model, comp_name, None)
                if comp:
                    try:
                        comp.load_state_dict(checkpoint[state_key])
                        print(f"  ✓ Loaded {comp_name}")
                    except Exception as e:
                        print(f"  ⚠ Failed to load {comp_name}: {e}")
    
    model.eval()
    print("✓ Model loaded")
    
    # Run evaluation
    print("\n[Step 3] Running evaluation...")
    
    from harpo.evaluation import HARPOMTv2Evaluator
    
    eval_cache_dir = os.path.join(args.output, "eval_cache")
    os.makedirs(eval_cache_dir, exist_ok=True)
    
    evaluator = HARPOMTv2Evaluator(model, device)
    results = evaluator.run_full_evaluation(
        test_sft, test_pref,
        max_samples=args.max_samples,
        cache_dir=eval_cache_dir
    )
    
    # Save results
    print("\n[Step 4] Saving results...")
    
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
        elif isinstance(obj, float) and obj != obj:  # NaN
            return None
        else:
            return obj
    
    from dataclasses import asdict
    with open(results_path, "w") as f:
        json.dump(serialize_for_json(asdict(results)), f, indent=2)
    
    print(f"✓ Results saved: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n📊 PRIMARY - Recommendation Quality:")
    print(f"  User Satisfaction:  {results.user_satisfaction:.4f}")
    print(f"  Engagement Score:   {results.engagement_score:.4f}")
    
    print("\n📈 RANKING METRICS:")
    print(f"  Recall@1:   {results.recall_at_1:.4f}")
    print(f"  Recall@10:  {results.recall_at_10:.4f}")
    print(f"  Recall@50:  {results.recall_at_50:.4f}")
    print(f"  MRR:        {results.mrr:.4f}")
    print(f"  NDCG@10:    {results.ndcg_at_10:.4f}")
    
    print("\n📝 GENERATION:")
    print(f"  BLEU-4:     {results.bleu_4:.4f}")
    print(f"  ROUGE-L:    {results.rouge_l:.4f}")
    print(f"  Distinct-2: {results.distinct_2:.4f}")
    
    print("\n🔧 VTO/TOOL:")
    print(f"  VTO F1:     {results.vto_f1:.4f}")
    
    print("\n🧠 REASONING (STAR):")
    print(f"  Depth:      {results.reasoning_depth:.2f}")
    print(f"  Quality:    {results.thought_quality:.4f}")
    
    return results


if __name__ == "__main__":
    main()