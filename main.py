# main.py

import json
import click
from pathlib import Path
from typing import Dict, Any

from identifier.core import LanguageIdentifier
from identifier.advanced import HybridIdentifier

# A helper to pretty-print our results dictionary
def display_results(result: Dict[str, Any]):
    """Formats and prints the identification results in a user-friendly way."""
    if "error" in result:
        click.secho(f"Error: {result['error']}", fg='red', bold=True)
        return

    prediction = result.get("prediction", "N/A")
    distribution = result.get("distribution", [])
    features = result.get("top_features", [])

    # Print a nice summary report 
    click.echo("\n" + "="*40)
    click.secho("# Language Identification Report", fg='yellow', bold=True)
    click.echo("="*40)

    click.echo(f"Top Prediction: ", nl=False)
    click.secho(f"{prediction.upper()}", fg='green', bold=True)
    
    click.echo("\n # Score Distribution (lower is better):")
    for item in distribution:
        lang = item.get('lang', '?').upper()
        score = item.get('score', 0.0)
        click.echo(f"  - {lang}: {score}")
        
    click.echo("\n # Top Features (most influential n-grams):")
    click.secho(f"  {', '.join(features)}", fg='cyan')
    click.echo("="*40)


@click.command()
@click.argument('text', type=str)
@click.option(
    '--model', '-m', # Added a short-form alias '-m'
    type=click.Choice(['simple', 'advanced'], case_sensitive=False),
    default='advanced', # Default is 'advanced' as it's the better model
    help='Choose model: "simple" (n-grams) or "advanced" (hybrid).'
)
@click.option('--alpha', type=click.FLOAT, default=0.5, help="Weight for hybrid model (advanced only).")
def cli(text: str, model: str, alpha: float):
    """
    Identifies the language of a given TEXT string using a selected model.
    
    Example usages:\n
    python main.py "This is a test sentence."\n
    python main.py "Ceci est une phrase de test." -m advanced --alpha 0.3
    """
    # This path logic is robust and finds the profiles folder correctly.
    profiles_path = Path(__file__).resolve().parent / "profiles"

    try:
        identifier = None
        if model == 'simple':
            click.secho("--- Using Simple Model (Character N-grams) ---", fg='blue')
            identifier = LanguageIdentifier(profile_dir=profiles_path)
        else: # model == 'advanced'
            click.secho(f"--- Using Advanced Hybrid Model (alpha={alpha}) ---", fg='magenta')
            identifier = HybridIdentifier(profile_dir=profiles_path, alpha=alpha)
        
        # Get the full results dictionary
        results = identifier.identify(text)
        
        # Use the helper to display it nicely
        display_results(results)

    except FileNotFoundError as e:
        click.secho(f"Error: Could not find language profiles.", fg='red', err=True)
        click.secho(f"Details: {e}", fg='red', err=True)
        click.secho("\n Please run 'python build_profiles.py' to generate them first.", err=True)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red', err=True)


if __name__ == '__main__':
    cli()