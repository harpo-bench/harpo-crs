"""
HARPO Command Line Interface

Usage:
    harpo evaluate outputs.json --metrics all
    harpo compare a.json b.json
    harpo explain context.txt response.txt
"""

import click
import json
from pathlib import Path
from typing import Optional
from .api import Evaluator, Comparator, Explainer


@click.group()
def cli():
    """HARPO: Hierarchical Agentic Reasoning with Preference Optimization"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--model", default=None, help="Path to pretrained model")
@click.option("--metrics", default="all", help="Metrics: all, quality, satisfaction")
@click.option("--output", "-o", default=None, help="Output file (JSON)")
@click.option("--confidence", is_flag=True, help="Include confidence scores")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def evaluate(input_file: str, model: Optional[str], metrics: str, 
             output: Optional[str], confidence: bool, device: str):
    """
    Evaluate responses from a JSON file.
    
    Input format:
        [{"context": "...", "response": "..."}, ...]
    """
    evaluator = Evaluator(model_path=model, device=device)
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    results = []
    for item in click.progressbar(data, label="Evaluating"):
        context = item.get("context", "")
        response = item.get("response", "")
        
        score = evaluator.score(context, response)
        result = score.to_dict()
        result["context"] = context[:100] + "..."
        result["response"] = response[:100] + "..."
        
        results.append(result)
    
    # Print summary
    click.echo("\n" + "="*50)
    click.echo("EVALUATION SUMMARY")
    click.echo("="*50)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    avg_diversity = sum(r["diversity"] for r in results) / len(results)
    avg_satisfaction = sum(r["satisfaction"] for r in results) / len(results)
    avg_engagement = sum(r["engagement"] for r in results) / len(results)
    
    click.echo(f"Avg Relevance:    {avg_relevance:.3f}")
    click.echo(f"Avg Diversity:    {avg_diversity:.3f}")
    click.echo(f"Avg Satisfaction: {avg_satisfaction:.3f}")
    click.echo(f"Avg Engagement:   {avg_engagement:.3f}")
    
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument("file_a", type=click.Path(exists=True))
@click.argument("file_b", type=click.Path(exists=True))
@click.option("--model", default=None, help="Path to pretrained model")
@click.option("--output", "-o", default=None, help="Output file (JSON)")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def compare(file_a: str, file_b: str, model: Optional[str], 
            output: Optional[str], device: str):
    """
    Compare two response files.
    
    File format:
        {"context": "...", "response_a": "...", "response_b": "..."}
    """
    comparator = Comparator(model_path=model, device=device)
    
    with open(file_a, "r") as f:
        data_a = json.load(f)
    with open(file_b, "r") as f:
        data_b = json.load(f)
    
    results = []
    
    for item_a, item_b in zip(data_a, data_b):
        context = item_a.get("context", "")
        response_a = item_a.get("response", "")
        response_b = item_b.get("response", "")
        
        comparison = comparator.compare(context, response_a, response_b)
        result = comparison.to_dict()
        result["context"] = context[:50] + "..."
        
        results.append(result)
    
    # Summary
    a_wins = sum(1 for r in results if r["preference_prob_a"] > 0.5)
    b_wins = sum(1 for r in results if r["preference_prob_b"] > 0.5)
    
    click.echo("\n" + "="*50)
    click.echo("COMPARISON SUMMARY")
    click.echo("="*50)
    click.echo(f"File A wins: {a_wins}/{len(results)}")
    click.echo(f"File B wins: {b_wins}/{len(results)}")
    avg_margin = sum(r["margin"] for r in results) / len(results)
    click.echo(f"Avg margin: {avg_margin:.3f}")
    
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument("context_file", type=click.Path(exists=True))
@click.argument("response_file", type=click.Path(exists=True))
@click.option("--model", default=None, help="Path to pretrained model")
@click.option("--output", "-o", default=None, help="Output file (JSON)")
@click.option("--device", default="cuda", help="Device: cuda or cpu")
def explain(context_file: str, response_file: str, model: Optional[str],
            output: Optional[str], device: str):
    """
    Explain evaluation of a response.
    
    Provides reasoning traces and signal analysis.
    """
    explainer = Explainer(model_path=model, device=device)
    
    with open(context_file, "r") as f:
        context = f.read()
    
    with open(response_file, "r") as f:
        response = f.read()
    
    explanation = explainer.explain(context, response)
    result = explanation.to_dict()
    
    click.echo("\n" + "="*50)
    click.echo("EXPLANATION")
    click.echo("="*50)
    
    click.echo("\nReasoning Trace:")
    for trace in result["reasoning_trace"]:
        click.echo(f"  - {trace}")
    
    click.echo("\nConfidence Signals:")
    for signal, value in result["confidence_signals"].items():
        click.echo(f"  {signal}: {value:.3f}")
    
    if result["weak_signals"]:
        click.echo("\nWeak Signals (0.3-0.7):")
        for signal in result["weak_signals"]:
            click.echo(f"  - {signal}")
    
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"\nExplanation saved to {output}")


if __name__ == "__main__":
    cli()
