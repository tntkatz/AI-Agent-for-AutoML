# AI-Agent-for-AutoML

Automated feature engineering and tabular model pipeline driven by an LLM loop. 

The tool generates baseline predictions, adds domain/statistical interaction features, prompts an LLM to propose new features in an iterative loop, validates them using out-of-fold CV score, and ensembles the final models.

## How it works

1. Runs a fast LightGBM baseline on raw data to establish the benchmark score.
2. Generates initial heuristic features (symmetric column pairs, group statistics, ratios).
3. Iteratively queries an LLM to generate `add_features(df)` Python functions:
   - Wraps generated features in AST-based `try/except` to prevent runtime crashes.
   - Evaluates the new feature set with 5-fold CV.
   - Keeps features only if the validation score strictly improves.
   - Drops features with zero gain importance.
4. Trains an ensemble (LightGBM, CatBoost, XGBoost) and blends predictions via Nelder-Mead optimization.
5. Saves the best verified predictions to `submission.csv`.

## Requirements

- Python 3.10+
- OpenAI-compatible API endpoint (Qwen, DeepSeek, etc.)

```bash
pip install -r requirements.txt
```

## Usage

Set environment variables for the LLM provider:

```bash
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="[https://api.your-provider.com/v1](https://api.your-provider.com/v1)"
export LLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
```

Run the pipeline by pointing it to a directory containing `train.csv`, `test.csv`, and `sample_submission.csv`:

```bash
python main.py /path/to/data_dir
```

Optional: provide `data_dictionary.csv` in the same directory to give the LLM column descriptions.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_ITERATIONS` | `15` | Maximum LLM feature generation iterations |
| `TASK_DIR` | `.` | Target data directory |
| `LLM_TIMEOUT` | `60` | Timeout per request (seconds) |

## License

MIT
