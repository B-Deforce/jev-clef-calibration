# Jev and Clef calibration experiment

**Live article:** [“To Jev, to Clef, or… to calibrate?”](https://bojedeforce.com/articles/to-jev-to-clef-or-to-calibrate.html)

This code asks Jev and Clef the same yes/no questions on 500 sentiment sentences and 500 redacted SMS messages, then checks the probabilities against the supplied labels.

This repository contains **code only**. You can download the public datasets and generate your own results with your own API access. Requests to both hosted services may incur charges.

## Original run and limits of reproduction

The original calls were made on **October 2, 2026**. The Jev request used `jev-latest`; all 1,000 saved Jev responses identified their model as `jev-1.13.0`. The Clef request and responses identified the model as `clef`; Cloudflare did not return an immutable revision ID. The exact prompts are in `run_predictions.py`. Future hosted weights or API behavior may change, so this code reproduces the **procedure**, not necessarily the original numerical results. Try `JEV_MODEL=jev-1.13.0` if TypeSafe still serves that version; otherwise the default is `jev-latest`. Inspect the `model` field in each output row before comparing runs.

The 500-row samples use fixed seeds. Sentiment sentences are deliberately clear positive or negative examples. The SMS sample keeps its class imbalance but replaces phone-like and email-like strings before either API call. Public benchmark data may have appeared in model training. Neither sample establishes production calibration.

## Requirements

- Python 3.9 or newer; all scripts use the standard library.
- Access to TypeSafe Jev and Cloudflare Workers AI Clef, with your own API credentials.
- The two public [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) and [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) ZIP files. UCI lists both as CC BY 4.0.

## Run the experiment

Run these commands from this folder:

```bash
python3 download_data.py
python3 prepare_data.py
cp .env.example .env.local
```

Fill in the three credential values in `.env.local` locally. `download_data.py` retrieves the original UCI ZIPs into ignored `data/raw/` and checks their SHA-256 hashes. If you download them manually, save them as `data/raw/sentiment_labelled_sentences.zip` and `data/raw/sms_spam_collection.zip` before running `prepare_data.py`. The prep script writes the deterministic samples and a manifest into ignored `data/processed/`.

Run four matched 500-row batches. The runner saves each completed row and resumes by ID after a network error:

```bash
python3 run_predictions.py jev data/processed/sentiment_pilot500.jsonl results/jev_sentiment_pilot500.jsonl
python3 run_predictions.py clef data/processed/sentiment_pilot500.jsonl results/clef_sentiment_pilot500.jsonl
python3 run_predictions.py jev data/processed/sms_spam_redacted_pilot500.jsonl results/jev_sms_redacted_pilot500.jsonl
python3 run_predictions.py clef data/processed/sms_spam_redacted_pilot500.jsonl results/clef_sms_redacted_pilot500.jsonl
```

Summarize each matched task:

```bash
python3 analyze_results.py data/processed/sentiment_pilot500.jsonl results/jev_sentiment_pilot500.jsonl results/clef_sentiment_pilot500.jsonl > results/sentiment_summary.json
python3 analyze_results.py data/processed/sms_spam_redacted_pilot500.jsonl results/jev_sms_redacted_pilot500.jsonl results/clef_sms_redacted_pilot500.jsonl > results/sms_summary.json
```

For a quick API check, `python3 try_clef.py` sends one synthetic message. You can also replace `pilot500` with `pilot100` in the preparation output and runner commands for a smaller preliminary run.

## What the analysis computes

`analyze_results.py` requires exactly matched row IDs from both models. It reports accuracy, positive-class precision and recall at the 0.5 threshold; Brier score and log loss; ten equal-width reliability bins and ECE; and ECCE-R from cumulative observed-minus-predicted residuals. Equal reported scores are grouped before evaluating the cumulative curve, so their arbitrary row order cannot change ECCE-R. This tie handling changes the original SMS Jev ECCE-R only from 0.1590 to 0.1589.

It also uses 2,000 **paired** bootstrap resamples, with seed `20261002`, to estimate central 95% intervals for Jev-minus-Clef differences in Brier and ECCE-R. A separate 1,000-draw simulation generates outcomes conditional on each model's reported probabilities to check whether its ECCE-R is unusual under exact calibration. These checks are exploratory for these samples; they do not account for dataset selection, prompt choice, label errors, or future model changes.

## Files

- `download_data.py`: fetch and verify the two UCI archives.
- `prepare_data.py`: deduplicate, redact SMS, and draw fixed-seed samples.
- `run_predictions.py`: call one provider on one prepared dataset and resume by row ID.
- `calibration_metrics.py`: scoring and calibrated-probability simulation.
- `analyze_results.py`: matched comparison, bootstrap intervals, and subgroup summaries.
- `try_clef.py`: optional one-request smoke test.

No `.env.local`, `data/`, or `results/` content belongs in a Git commit.
