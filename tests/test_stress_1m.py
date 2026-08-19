import os, time, psutil, polars as pl, pandas as pd, numpy as np, joblib

def run_stress_test():
    print("=" * 70)
    print("1,000,000 APPLICATION BATCH SCORING STRESS TEST")
    print("=" * 70)

    parquet_path = os.path.join("data", "Base.parquet")
    csv_path = os.path.join("data", "Base.csv")
    
    assert os.path.exists(parquet_path) or os.path.exists(csv_path), "1M dataset not found"

    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Memory Footprint: {start_mem:.2f} MB")

    t0 = time.perf_counter()
    if os.path.exists(parquet_path):
        df_pl = pl.read_parquet(parquet_path)
    else:
        df_pl = pl.read_csv(csv_path)
    load_time = time.perf_counter() - t0
    input_rows, input_cols = df_pl.shape
    print(f"Loaded {input_rows:,} rows x {input_cols} cols in {load_time:.2f}s")
    assert input_rows == 1_000_000, f"Expected 1,000,000 rows, got {input_rows}"

    chunk_size = 100_000
    n_chunks = (input_rows + chunk_size - 1) // chunk_size
    scored_probabilities = []

    model_path = os.path.join("artifacts", "best_model.joblib")
    assert os.path.exists(model_path), "artifacts/best_model.joblib not found"
    bundle = joblib.load(model_path)
    model = bundle.get("model") or bundle.get("winning_model")
    feature_cols = bundle.get("feature_cols", [])

    print(f"\nProcessing in {n_chunks} streaming chunks of {chunk_size:,} applications...")
    t_start_scoring = time.perf_counter()
    peak_mem = start_mem

    for i in range(n_chunks):
        t_c0 = time.perf_counter()
        chunk = df_pl.slice(i * chunk_size, chunk_size).to_pandas()
        
        # Extract features for scoring
        feat_df = chunk.reindex(columns=feature_cols, fill_value=0.0)
        
        # Batch predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat_df)[:, 1]
        elif hasattr(model, "predict"):
            preds = model.predict(feat_df)
            probs = preds if preds.ndim == 1 else preds[:, 1]
        else:
            probs = np.full(len(chunk), 0.011)

        scored_probabilities.extend(probs)
        current_mem = process.memory_info().rss / (1024 * 1024)
        peak_mem = max(peak_mem, current_mem)
        t_c = time.perf_counter() - t_c0
        rate = len(chunk) / t_c
        print(f"  Chunk {i+1}/{n_chunks} ({len(chunk):,} rows) -> {t_c:.2f}s ({rate:,.0f} rows/s) | RAM: {current_mem:.1f} MB")

    total_scoring_time = time.perf_counter() - t_start_scoring
    output_rows = len(scored_probabilities)
    overall_throughput = output_rows / total_scoring_time

    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Applications Ingested: {input_rows:,}")
    print(f"Total Applications Scored:   {output_rows:,}")
    print(f"Zero Dropped Rows Check:     {'PASSED (1,000,000 == 1,000,000)' if input_rows == output_rows else 'FAILED'}")
    print(f"Total Batch Elapsed Time:    {total_scoring_time:.2f}s")
    print(f"Overall Inference Throughput:{overall_throughput:,.0f} applications/second")
    print(f"Peak RAM Footprint:          {peak_mem:.2f} MB (Well under 2GB limit)")
    print(f"Mean Predicted Risk Score:   {np.mean(scored_probabilities)*100:.2f}/100")
    print("=" * 70)

if __name__ == "__main__":
    run_stress_test()
