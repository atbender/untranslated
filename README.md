## Translation Back-Translation Leveler (PoC)

A Python web app that degrades translations through deterministic multi-hop back-translation using OpenRouter API.

- Uses OpenRouter API for real translations
- Deterministic language chain selection: same seed + hops => same language sequence
- Three built-in examples (Spanish, French, German) with English references

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get an OpenRouter API key from https://openrouter.ai/

3. Set the API key as an environment variable:
```bash
export OPENROUTER_API_KEY='your-api-key-here'
```

### Run

```bash
python3 app.py
```

Then open `http://127.0.0.1:8000` in your browser.

### How it works

- The slider controls the number of "hops" (0-10), which determines how many round-trip translations are performed
- Each hop translates: English → target language → English
- The language chain is deterministically selected from a pool of 10 languages based on the seed and example ID
- Same seed + hops yields the same language chain (though API translations may have slight variations)
- Language pool: Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Polish, Turkish, Japanese

### Example

- **Hops = 0**: No translation, returns original reference
- **Hops = 1**: English → Spanish → English (one round-trip)
- **Hops = 2**: English → Spanish → English → French → English (two round-trips)
- And so on...

Each additional hop increases degradation as translation errors accumulate.
