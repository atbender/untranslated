#!/usr/bin/env python3
import json
import os
import sys
from hashlib import sha256
from typing import List, Tuple, Dict, Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

try:
	import requests
except ImportError:
	print("Error: 'requests' library is required. Install it with: pip install requests")
	sys.exit(1)

try:
	from dotenv import load_dotenv
	load_dotenv()  # Load environment variables from .env file
except ImportError:
	print("Warning: 'python-dotenv' library not found. Install it with: pip install python-dotenv")
	print("Continuing without .env file support...")


# Example sentences with their high-quality English translations
EXAMPLES = [
	{
		"id": 1,
		"language": "Spanish",
		"source": "¿Dónde está la biblioteca?",
		"reference": "Where is the library?",
	},
	{
		"id": 2,
		"language": "French",
		"source": "Je vous remercie pour votre aide précieuse.",
		"reference": "I thank you for your valuable help.",
	},
	{
		"id": 3,
		"language": "German",
		"source": "Können Sie mir bitte den Weg zum Bahnhof zeigen?",
		"reference": "Could you please show me the way to the train station?",
	},
	{
		"id": 4,
		"language": "Technical",
		"source": "",
		"reference": "The temporal sequence of reasoning in Large Language Models is the defining variable of their cognitive architecture. The evidence conclusively demonstrates that Reasoning-Before-Answer is the only reliable mechanism for generating accurate solutions to complex problems, as it respects the causal, autoregressive nature of the Transformer. It allows the model to \"think\" by using token generation as a form of working memory.",
	},
]

# Fixed pool of diverse languages including low-resource languages (codes with human names)
LANG_POOL = [
	("es", "Spanish"),
	("fr", "French"),
	("de", "German"),
	("it", "Italian"),
	("pt", "Portuguese"),
	("nl", "Dutch"),
	("sv", "Swedish"),
	("pl", "Polish"),
	("tr", "Turkish"),
	("ja", "Japanese"),
	("zh", "Chinese"),
	("ar", "Arabic"),
	("hi", "Hindi"),
	("th", "Thai"),
	("vi", "Vietnamese"),
	("sw", "Swahili"),
	("am", "Amharic"),
	("zu", "Zulu"),
	("id", "Indonesian"),
	("uk", "Ukrainian"),
	("he", "Hebrew"),
	("ko", "Korean"),
	("bn", "Bengali"),
	("ta", "Tamil"),
	("te", "Telugu"),
]

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"  # Using Gemini Flash Lite for PoC

# Translation cache: stores results to avoid redundant API calls
_translation_cache: Dict[str, str] = {}  # Key: (text, source_lang, target_lang), Value: translated_text
_roundtrip_cache: Dict[str, str] = {}  # Key: (text, lang_code), Value: roundtrip_result
_degradation_cache: Dict[str, Tuple[str, List[str]]] = {}  # Key: (reference, hops, seed, example_id), Value: (degraded, chain)

# Custom examples storage (in-memory, resets on server restart)
_custom_examples: Dict[int, Dict[str, Any]] = {}  # Key: example_id, Value: example dict
_next_custom_id = 1000  # Start custom IDs high to avoid conflicts with built-in examples


def _hash_to_unit(s: str) -> float:
	"""
	Deterministically map a string to [0,1) for language chain selection.
	"""
	h = sha256(s.encode("utf-8")).digest()
	n = int.from_bytes(h[:8], "big")
	return (n % (10**8)) / float(10**8)


def _language_chain(seed: str, example_id: int, hops: int) -> List[str]:
	"""
	Choose a deterministic sequence of language codes of length 'hops'
	by selecting diverse languages from the pool based on seed and example id.
	Uses hash-based selection to ensure diversity rather than just sequential selection.
	"""
	if hops <= 0:
		return []
	
	codes = []
	used_indices = set()
	
	for i in range(hops):
		# Create a unique hash for each hop position to ensure diversity
		hash_input = f"chain|{seed}|{example_id}|{i}"
		hash_val = _hash_to_unit(hash_input)
		
		# Try to find an unused language index
		attempts = 0
		while attempts < len(LANG_POOL):
			idx = int(hash_val * len(LANG_POOL))
			if idx not in used_indices:
				used_indices.add(idx)
				codes.append(LANG_POOL[idx][0])
				break
			# If already used, modify hash slightly
			hash_val = (hash_val + 0.1) % 1.0
			attempts += 1
		
		# Fallback: if we've used all languages, allow reuse
		if attempts >= len(LANG_POOL):
			idx = int(hash_val * len(LANG_POOL))
			codes.append(LANG_POOL[idx][0])
	
	return codes


def translate_with_openrouter(text: str, source_lang: str, target_lang: str, api_key: str) -> str:
	"""
	Translate text using OpenRouter API.
	Returns the translated text or raises an exception on error.
	Uses cache to avoid redundant API calls.
	"""
	# Check cache first
	cache_key = f"{text}|{source_lang}|{target_lang}"
	if cache_key in _translation_cache:
		return _translation_cache[cache_key]
	
	# Get language names for better prompts
	if source_lang == "en":
		source_name = "English"
	else:
		source_name = next((name for code, name in LANG_POOL if code == source_lang), source_lang.upper())
	
	if target_lang == "en":
		target_name = "English"
	else:
		target_name = next((name for code, name in LANG_POOL if code == target_lang), target_lang.upper())

	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
		"HTTP-Referer": "http://localhost:8000",  # Optional: for OpenRouter analytics
	}

	payload = {
		"model": OPENROUTER_MODEL,
		"messages": [
			{
				"role": "system",
				"content": f"You are a STRICT translator. Translate the following text from {source_name} to {target_name} LITERALLY and STRICTLY as it. Only output the translation, nothing else.",
			},
			{"role": "user", "content": text},
		],
		"temperature": 0.3,  # Lower temperature for more deterministic translations
		"max_tokens": 500,
	}

	try:
		response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
		response.raise_for_status()
		data = response.json()
		
		if "choices" not in data or not data["choices"]:
			raise ValueError("No translation returned from API")
		
		translated = data["choices"][0]["message"]["content"].strip()
		# Store in cache
		_translation_cache[cache_key] = translated
		return translated
	except requests.exceptions.RequestException as e:
		raise Exception(f"OpenRouter API error: {e}")
	except (KeyError, IndexError) as e:
		raise Exception(f"Unexpected API response format: {e}")


def backtranslate_roundtrip(text: str, lang_code: str, api_key: str) -> str:
	"""
	Perform a round-trip translation: English -> target language -> English.
	Uses cache to avoid redundant API calls.
	"""
	# Check cache first
	cache_key = f"{text}|{lang_code}"
	if cache_key in _roundtrip_cache:
		return _roundtrip_cache[cache_key]
	
	# Translate to target language
	translated = translate_with_openrouter(text, "en", lang_code, api_key)
	# Translate back to English
	back_translated = translate_with_openrouter(translated, lang_code, "en", api_key)
	
	# Store in cache
	_roundtrip_cache[cache_key] = back_translated
	return back_translated


def degrade_via_backtranslation(reference: str, hops: int, seed: str, example_id: int, api_key: str) -> Tuple[str, List[str]]:
	"""
	Apply 'hops' round-trips through a deterministic chain of languages using OpenRouter.
	Returns final degraded English and the chain of language codes used.
	Uses cache to avoid redundant API calls.
	"""
	hops = max(0, int(hops))
	
	# Check cache first
	cache_key = f"{reference}|{hops}|{seed}|{example_id}"
	if cache_key in _degradation_cache:
		return _degradation_cache[cache_key]
	
	chain = _language_chain(seed, example_id, hops)
	out = reference
	
	for i, code in enumerate(chain):
		try:
			out = backtranslate_roundtrip(out, code, api_key)
		except Exception as e:
			# If translation fails, return partial result with error info
			error_msg = f"[Translation error at hop {i+1}: {str(e)}]"
			result = (f"{out} {error_msg}", chain)
			_degradation_cache[cache_key] = result
			return result
	
	result = (out, chain)
	_degradation_cache[cache_key] = result
	return result


def render_index_html() -> bytes:
	html = """<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Translation Back-Translation Leveler (PoC)</title>
	<style>
		body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }
		h1 { font-size: 20px; margin: 0 0 16px; }
		.controls { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }
		.slider-wrap { display: flex; align-items: center; gap: 8px; }
		input[type="range"] { width: 280px; }
		input[type="text"] { padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; }
		table { width: 100%; border-collapse: collapse; }
		th, td { padding: 10px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
		th { text-align: left; background: #fafafa; }
		td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #333; }
		.badge { display: inline-block; padding: 2px 6px; font-size: 12px; background:#eef; border:1px solid #dde; border-radius: 4px; }
		.badge.custom { background:#fee; }
		.ref { color:#666; }
		.error { color: #c33; }
		.footer { margin-top: 16px; color: #666; font-size: 13px; }
		.loading { color: #666; font-style: italic; }
		.add-custom { margin-top: 20px; padding: 16px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd; }
		.add-custom input { width: 100%; max-width: 600px; padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px; margin-right: 8px; }
		.add-custom button { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 6px; cursor: pointer; }
		.add-custom button:hover { background: #0056b3; }
		.add-custom .input-group { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
	</style>
	<script>
		const fetchDegraded = async (hops, seed) => {
			const url = `/api/degrade?hops=${encodeURIComponent(hops)}&seed=${encodeURIComponent(seed)}`;
			const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
			if (!res.ok) {
				const error = await res.json().catch(() => ({ error: 'Request failed' }));
				throw new Error(error.error || 'Request failed');
			}
			return await res.json();
		};

		let debounceTimer = null;
		const debounce = (fn, delay = 300) => {
			return (...args) => {
				clearTimeout(debounceTimer);
				debounceTimer = setTimeout(() => fn(...args), delay);
			};
		};

		const updateUI = (data) => {
			const tbody = document.querySelector('#examples-body');
			tbody.innerHTML = '';
			for (const ex of data.examples) {
				const tr = document.createElement('tr');
				const isCustom = ex.is_custom || false;
				const badgeClass = isCustom ? 'badge custom' : 'badge';
				const languageLabel = isCustom ? 'Custom' : ex.language;
				tr.innerHTML = `
					<td><span class="${badgeClass}">${languageLabel}</span></td>
					<td class="mono">${ex.source || '—'}</td>
					<td class="ref">${ex.reference}</td>
					<td class="mono">
						<div>${ex.degraded}</div>
						<div class="ref">Chain: ${ex.chain.join(' → ') || '—'}</div>
					</td>
				`;
				tbody.appendChild(tr);
			}
		};

		const addCustomExample = async () => {
			const input = document.querySelector('#custom-text');
			const text = input.value.trim();
			if (!text) {
				alert('Please enter some text');
				return;
			}

			try {
				const res = await fetch('/api/add-example', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ text: text })
				});
				if (!res.ok) {
					const error = await res.json().catch(() => ({ error: 'Request failed' }));
					throw new Error(error.error || 'Failed to add example');
				}
				input.value = '';
				refresh(); // Refresh to show the new example
			} catch (e) {
				alert('Error: ' + e.message);
				console.error(e);
			}
		};

		const refresh = async () => {
			const hops = document.querySelector('#hops').value;
			const seed = document.querySelector('#seed').value || 'seed';
			document.querySelector('#hops-value').textContent = hops;
			const tbody = document.querySelector('#examples-body');
			tbody.innerHTML = '<tr><td colspan="4" class="loading">Translating...</td></tr>';
			try {
				const data = await fetchDegraded(hops, seed);
				updateUI(data);
			} catch (e) {
				tbody.innerHTML = `<tr><td colspan="4" class="error">Error: ${e.message}</td></tr>`;
				console.error(e);
			}
		};

		document.addEventListener('DOMContentLoaded', () => {
			document.querySelector('#hops').addEventListener('input', debounce(refresh, 500));
			document.querySelector('#seed').addEventListener('input', debounce(refresh, 500));
			document.querySelector('#add-custom-btn').addEventListener('click', addCustomExample);
			document.querySelector('#custom-text').addEventListener('keypress', (e) => {
				if (e.key === 'Enter') {
					addCustomExample();
				}
			});
			refresh();
		});
	</script>
</head>
<body>
	<h1>Back-Translation Leveler (PoC)</h1>
	<div class="controls">
		<div class="slider-wrap">
			<label for="hops">Hops:</label>
			<input id="hops" type="range" min="0" max="15" step="1" value="2" />
			<strong id="hops-value">2</strong>
		</div>
		<div>
			<label for="seed">Seed:</label>
			<input id="seed" type="text" value="poc-seed" />
		</div>
	</div>
	<table>
		<thead>
			<tr>
				<th>Language</th>
				<th>Source</th>
				<th>Reference (good)</th>
				<th>Degraded (via back-translation)</th>
			</tr>
		</thead>
		<tbody id="examples-body"></tbody>
	</table>
	<div class="add-custom">
		<div class="input-group">
			<input id="custom-text" type="text" placeholder="Enter custom text to test mistranslation..." />
			<button id="add-custom-btn">Add Custom Example</button>
		</div>
		<div class="footer" style="margin-top: 8px;">
			Add your own text to see how it degrades through back-translation. The text will be used as the reference translation.
		</div>
	</div>
	<div class="footer">
		Adjust hops to increase the number of deterministic back-translation round-trips across a fixed language pool. Same seed + hops yields identical language chain (but translations may vary slightly due to API non-determinism).
	</div>
</body>
</html>
"""
	return html.encode("utf-8")


def handle_degrade_api(params: Dict[str, List[str]]) -> Dict[str, Any]:
	"""
	Handle the /api/degrade endpoint.
	"""
	hops_str = params.get("hops", [None])[0]
	seed = params.get("seed", ["poc-seed"])[0]
	
	hops = 2
	if hops_str is not None:
		try:
			hops = max(0, min(25, int(hops_str)))  # Increased max to match language pool size
		except ValueError:
			hops = 2

	# Get API key from environment variable
	api_key = os.getenv("OPENROUTER_API_KEY")
	if not api_key:
		return {
			"error": "OPENROUTER_API_KEY environment variable not set. Please set it before running the app.",
			"examples": [],
			"hops": hops,
			"seed": seed,
		}

	examples_out = []
	# Process built-in examples
	for ex in EXAMPLES:
		try:
			degraded, chain = degrade_via_backtranslation(ex["reference"], hops, seed, ex["id"], api_key)
			examples_out.append(
				{
					"id": ex["id"],
					"language": ex["language"],
					"source": ex["source"],
					"reference": ex["reference"],
					"degraded": degraded,
					"chain": chain,
					"is_custom": False,
				}
			)
		except Exception as e:
			examples_out.append(
				{
					"id": ex["id"],
					"language": ex["language"],
					"source": ex["source"],
					"reference": ex["reference"],
					"degraded": f"[Error: {str(e)}]",
					"chain": [],
					"is_custom": False,
				}
			)
	
	# Process custom examples
	for ex_id, ex in _custom_examples.items():
		try:
			degraded, chain = degrade_via_backtranslation(ex["reference"], hops, seed, ex_id, api_key)
			examples_out.append(
				{
					"id": ex_id,
					"language": "Custom",
					"source": ex.get("source", ""),
					"reference": ex["reference"],
					"degraded": degraded,
					"chain": chain,
					"is_custom": True,
				}
			)
		except Exception as e:
			examples_out.append(
				{
					"id": ex_id,
					"language": "Custom",
					"source": ex.get("source", ""),
					"reference": ex["reference"],
					"degraded": f"[Error: {str(e)}]",
					"chain": [],
					"is_custom": True,
				}
			)

	return {"examples": examples_out, "hops": hops, "seed": seed}


def handle_add_example_api(body_data: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Handle the /api/add-example endpoint.
	Adds a custom example to the in-memory storage.
	"""
	global _next_custom_id
	
	text = body_data.get("text", "").strip()
	if not text:
		return {"error": "Text is required", "success": False}
	
	# Create new custom example
	example_id = _next_custom_id
	_next_custom_id += 1
	
	_custom_examples[example_id] = {
		"id": example_id,
		"reference": text,
		"source": "",  # No source language for custom examples
	}
	
	return {
		"success": True,
		"id": example_id,
		"message": f"Custom example added with ID {example_id}",
	}


def app(environ, start_response):
	path = environ.get("PATH_INFO", "/")
	method = environ.get("REQUEST_METHOD", "GET").upper()

	# Route: index
	if method == "GET" and path == "/":
		body = render_index_html()
		start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
		return [body]

	# Route: API degrade
	if method == "GET" and path == "/api/degrade":
		params = parse_qs(environ.get("QUERY_STRING", ""))
		try:
			data = handle_degrade_api(params)
			body = json.dumps(data, ensure_ascii=False).encode("utf-8")
			status = "200 OK"
		except Exception as e:
			error_data = {"error": str(e)}
			body = json.dumps(error_data, ensure_ascii=False).encode("utf-8")
			status = "500 Internal Server Error"
		
		start_response(status, [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))])
		return [body]

	# Route: API add example
	if method == "POST" and path == "/api/add-example":
		try:
			# Read request body
			content_length = int(environ.get("CONTENT_LENGTH", 0))
			request_body = environ["wsgi.input"].read(content_length).decode("utf-8")
			body_data = json.loads(request_body) if request_body else {}
			
			data = handle_add_example_api(body_data)
			body = json.dumps(data, ensure_ascii=False).encode("utf-8")
			status = "200 OK"
		except json.JSONDecodeError:
			error_data = {"error": "Invalid JSON in request body", "success": False}
			body = json.dumps(error_data, ensure_ascii=False).encode("utf-8")
			status = "400 Bad Request"
		except Exception as e:
			error_data = {"error": str(e), "success": False}
			body = json.dumps(error_data, ensure_ascii=False).encode("utf-8")
			status = "500 Internal Server Error"
		
		start_response(status, [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))])
		return [body]

	# Fallback 404
	msg = b"Not Found"
	start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(msg)))])
	return [msg]


def main(argv: List[str]) -> int:
	# Check for API key
	if not os.getenv("OPENROUTER_API_KEY"):
		print("Warning: OPENROUTER_API_KEY environment variable not set.")
		print("Please set it before running the app:")
		print("  export OPENROUTER_API_KEY='your-api-key-here'")
		print("\nYou can get an API key from https://openrouter.ai/")
		print("\nContinuing anyway, but translations will fail...\n")

	host = "127.0.0.1"
	port = 8000
	with make_server(host, port, app) as httpd:
		print(f"Serving on http://{host}:{port}  (Press Ctrl+C to stop)")
		try:
			httpd.serve_forever()
		except KeyboardInterrupt:
			print("\nShutting down...")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv))
