#!/usr/bin/env python3
"""
Generate a BLEU score graph showing translation quality degradation across different hop values.
"""
import json
import os
import sys
import requests
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

try:
	from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
	print("Error: 'nltk' library is required. Install it with: pip install nltk")
	print("After installing, you may need to download data: python -c 'import nltk; nltk.download(\"punkt\")'")
	sys.exit(1)

try:
	from dotenv import load_dotenv
	load_dotenv()
except ImportError:
	pass


# API endpoint
API_URL = "http://127.0.0.1:8000/api/degrade"
SEED = "bleu-graph-seed"

# Example to test (using the technical text)
TEST_REFERENCE = "The temporal sequence of reasoning in Large Language Models is the defining variable of their cognitive architecture. The evidence conclusively demonstrates that Reasoning-Before-Answer is the only reliable mechanism for generating accurate solutions to complex problems, as it respects the causal, autoregressive nature of the Transformer. It allows the model to \"think\" by using token generation as a form of working memory."


def get_degraded_translation(hops: int, seed: str = SEED) -> str:
	"""
	Call the API to get degraded translation for a given number of hops.
	"""
	params = {
		"hops": hops,
		"seed": seed,
	}
	
	try:
		response = requests.get(API_URL, params=params, timeout=120)
		response.raise_for_status()
		data = response.json()
		
		# Find the technical example (id=4) in the results
		for ex in data.get("examples", []):
			if ex.get("id") == 4:  # Technical example
				return ex.get("degraded", "")
		
		# Fallback: use first example if technical not found
		if data.get("examples"):
			return data["examples"][0].get("degraded", "")
		
		return ""
	except Exception as e:
		print(f"Error fetching translation for hops={hops}: {e}")
		return ""


def calculate_bleu(reference: str, candidate: str) -> float:
	"""
	Calculate BLEU score between reference and candidate translation.
	"""
	# Tokenize sentences
	reference_tokens = reference.lower().split()
	candidate_tokens = candidate.lower().split()
	
	if len(candidate_tokens) == 0:
		return 0.0
	
	# Use smoothing function to handle cases where n-grams don't match
	smoothing = SmoothingFunction().method1
	
	try:
		score = sentence_bleu(
			[reference_tokens],
			candidate_tokens,
			smoothing_function=smoothing
		)
		return score
	except Exception as e:
		print(f"Error calculating BLEU: {e}")
		return 0.0


def generate_bleu_graph(max_hops: int = 20, output_file: str = "bleu_scores.png"):
	"""
	Generate BLEU score graph across different hop values.
	"""
	print(f"Generating BLEU score graph for hops 0-{max_hops}...")
	print("This may take a while as it needs to make API calls for each hop value.")
	print(f"Testing with reference text: {TEST_REFERENCE[:80]}...")
	print()
	
	hops_values = list(range(0, max_hops + 1))
	bleu_scores = []
	degraded_texts = []
	
	for hops in hops_values:
		print(f"Testing hops={hops}...", end=" ", flush=True)
		
		if hops == 0:
			# Hops=0 means no translation, so it should be identical
			degraded = TEST_REFERENCE
		else:
			degraded = get_degraded_translation(hops, SEED)
		
		if not degraded:
			print("FAILED - no translation returned")
			bleu_scores.append(0.0)
			degraded_texts.append("")
			continue
		
		degraded_texts.append(degraded)
		bleu = calculate_bleu(TEST_REFERENCE, degraded)
		bleu_scores.append(bleu)
		print(f"BLEU={bleu:.4f}")
	
	# Create the graph
	plt.figure(figsize=(12, 6))
	plt.plot(hops_values, bleu_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
	plt.xlabel('Number of Hops (Translation Round-trips)', fontsize=12)
	plt.ylabel('BLEU Score', fontsize=12)
	plt.title('Translation Quality Degradation: BLEU Score vs Number of Hops', fontsize=14, fontweight='bold')
	plt.grid(True, alpha=0.3)
	plt.xlim(-0.5, max_hops + 0.5)
	plt.ylim(0, max(1.0, max(bleu_scores) * 1.1))
	
	# Add annotations for key points
	if len(bleu_scores) > 0:
		plt.axhline(y=bleu_scores[0], color='g', linestyle='--', alpha=0.5, label=f'Initial BLEU: {bleu_scores[0]:.4f}')
		if len(bleu_scores) > 1:
			plt.axhline(y=bleu_scores[-1], color='r', linestyle='--', alpha=0.5, label=f'Final BLEU: {bleu_scores[-1]:.4f}')
	
	plt.legend()
	plt.tight_layout()
	
	# Save the graph
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"\nGraph saved to: {output_file}")
	
	# Print summary statistics
	print("\nSummary Statistics:")
	print(f"  Initial BLEU (hops=0): {bleu_scores[0]:.4f}")
	print(f"  Final BLEU (hops={max_hops}): {bleu_scores[-1]:.4f}")
	print(f"  Degradation: {((bleu_scores[0] - bleu_scores[-1]) / bleu_scores[0] * 100):.1f}%")
	print(f"  Average BLEU: {np.mean(bleu_scores):.4f}")
	print(f"  Min BLEU: {min(bleu_scores):.4f} (at hops={hops_values[bleu_scores.index(min(bleu_scores))]})")
	print(f"  Max BLEU: {max(bleu_scores):.4f} (at hops={hops_values[bleu_scores.index(max(bleu_scores))]})")
	
	# Show some example degraded texts
	print("\nExample Degraded Texts:")
	for i in [0, max_hops // 4, max_hops // 2, max_hops]:
		if i < len(degraded_texts) and degraded_texts[i]:
			print(f"\nHops={i}, BLEU={bleu_scores[i]:.4f}:")
			print(f"  {degraded_texts[i][:150]}...")


def main():
	"""
	Main function.
	"""
	# Check if server is running
	try:
		response = requests.get("http://127.0.0.1:8000/", timeout=5)
		if response.status_code != 200:
			print("Warning: Server might not be running correctly")
	except Exception as e:
		print(f"Error: Cannot connect to server at http://127.0.0.1:8000")
		print(f"Please make sure the app.py server is running.")
		print(f"Error: {e}")
		sys.exit(1)
	
	# Generate graph
	generate_bleu_graph(max_hops=20, output_file="bleu_scores.png")
	print("\nDone!")


if __name__ == "__main__":
	main()

