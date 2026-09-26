#!/usr/bin/env python3
"""
HARPO: Interactive Chat Demo

Usage:
    python scripts/chat.py --checkpoint outputs/checkpoints/final
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch


def main():
    parser = argparse.ArgumentParser(description="HARPO Interactive Chat")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--domain", default="movies", 
                        choices=["movies", "books", "fashion", "electronics", "food", "general"],
                        help="Recommendation domain")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HARPO: Interactive Recommendation Chat")
    print("=" * 60)
    
    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Load model
    print("\nLoading model...")
    
    base_model_path = os.path.join(args.checkpoint, "base_model")
    components_path = os.path.join(args.checkpoint, "components.pt")
    
    if not os.path.exists(base_model_path):
        print(f"ERROR: base_model not found: {base_model_path}")
        sys.exit(1)
    
    from harpo.config import ModelConfig, TrainingConfig
    from harpo.model import HARPOMTv2
    
    model_config = ModelConfig(model_name=base_model_path)
    training_config = TrainingConfig()
    
    model = HARPOMTv2(model_config, training_config)
    model.load_base_model(device)
    
    # Load components if available
    if os.path.exists(components_path):
        checkpoint = torch.load(components_path, map_location=device)
        
        for comp_name, state_key in [
            ("bridge", "bridge_state_dict"),
            ("star", "star_state_dict"),
            ("charm", "charm_state_dict"),
            ("maven", "maven_state_dict"),
            ("vto_head", "vto_head_state_dict"),
        ]:
            if state_key in checkpoint:
                comp = getattr(model, comp_name, None)
                if comp:
                    try:
                        comp.load_state_dict(checkpoint[state_key])
                    except:
                        pass
    
    model.eval()
    print("✓ Model loaded!\n")
    
    print(f"Domain: {args.domain}")
    print("Type 'quit' to exit, 'clear' to reset conversation\n")
    print("-" * 40)
    
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                history = []
                print("Conversation cleared.")
                continue
            
            if not user_input:
                continue
            
            # Build prompt
            history.append(f"User: {user_input}")
            
            # Format: domain tag + recent history
            prompt = f"<|domain:{args.domain}|>\n\n"
            prompt += "\n".join(history[-6:])  # Last 3 turns
            prompt += "\nAssistant: "
            
            # Generate
            inputs = model.tokenizer(
                prompt, return_tensors="pt",
                max_length=512, truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            
            # Decode response
            response = model.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up
            if "\nUser:" in response:
                response = response.split("\nUser:")[0].strip()
            
            # Remove thinking tags for display
            import re
            clean_response = re.sub(r'<\|think\|>.*?<\|/think\|>\s*', '', response)
            
            # Show thinking if present
            think_match = re.search(r'<\|think\|>(.*?)<\|/think\|>', response)
            if think_match:
                vtos = think_match.group(1)
                print(f"\n[Reasoning: {vtos}]")
            
            print(f"\nAssistant: {clean_response}")
            
            history.append(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()