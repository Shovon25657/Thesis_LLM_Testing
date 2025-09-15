from llama_cpp import Llama
import pandas as pd
import re
import time
import os

# ============= CONFIG =============
MODEL_PATH = "./Phi-3-mini-4k-instruct.Q4_0.gguf"
CSV_PATH = "./200_synthetic_iov_tasks.csv"
OUTPUT_PATH = "./Phi-3(Priority_scores).csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {CSV_PATH}")

# ============= LOAD MODEL =============
print("🧠 Loading Phi-3-mini (CPU, 8GB RAM optimized)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    n_batch=256,
    verbose=False
)
print("✅ Model loaded.")


def build_prompt(row):
    # Extract features
    task_type = row['task_type']
    criticality = row['criticality']
    data_size_kb = row['data_size_kb']
    deadline_ms = row['deadline_ms']
    vehicle_speed = row['vehicle_speed_kmph']
    location = row['current_location']
    network = row['network_condition']
    distance_km = row['edge_node_distance_km']
    comp_flops = row['computation_intensity_flops']
    cost_pref = row['cost_weight_preference']
    time_of_day = row['time_of_day']
    mobility = row['mobility_pattern']
    context_text = row['context_text']

    # ✅ REVISED PROMPT — Optimized for Phi-3-mini
    prompt = f"""<|system|>
You are an AI priority engine for Internet of Vehicles (IoV). Your job is to assign a priority score from 0.000 to 1.000 based on task context.

Use this logic:
- Safety + High Speed + Tight Deadline (≤100ms) → Score ≥ 0.90
- Safety + Low Speed or Loose Deadline → Score 0.70–0.89
- V2X Communication + Congested Network → Score 0.60–0.85
- Diagnostics + Off-Peak + Cost-Sensitive → Score 0.30–0.55
- Navigation + Good Network + Medium Deadline → Score 0.40–0.65
- Infotainment + Vehicle Stationary + High Data Size → Score ≤ 0.25

Consider:
- Higher vehicle speed → higher urgency
- Congested network → higher score if latency-sensitive
- Cost-sensitive + non-safety → lower score
- Off-peak hours → slightly lower score for non-critical tasks

Output ONLY the score as a number like 0.753 — nothing else.<|end|>
<|user|>
Task Type: Safety
Criticality: High
Deadline: 50 ms
Vehicle Speed: 98 km/h
Network: Congested
Cost Preference: Latency-Oriented
Context: Emergency braking needed at high speed on highway.
Score:<|end|>
<|assistant|>
0.987<|end|>
<|user|>
Task Type: Infotainment
Criticality: Low
Deadline: 860 ms
Vehicle Speed: 0 km/h
Network: Congested
Cost Preference: Cost-Sensitive
Context: Music streaming, vehicle stationary in urban area.
Score:<|end|>
<|assistant|>
0.125<|end|>
<|user|>
Task Type: Diagnostics
Criticality: Medium
Deadline: 274 ms
Vehicle Speed: 87 km/h
Network: Good
Cost Preference: Balanced
Context: Predictive maintenance while moving in urban area.
Score:<|end|>
<|assistant|>
0.543<|end|>
<|user|>
Task Type: {task_type}
Criticality: {criticality}
Deadline: {deadline_ms} ms
Vehicle Speed: {vehicle_speed} km/h
Network: {network}
Cost Preference: {cost_pref}
Context: {context_text}
Score:<|end|>
<|assistant|>
"""
    return prompt


def get_priority_score(prompt):
    try:
        output = llm(
            prompt,
            max_tokens=15,
            temperature=0.3,
            top_p=0.95,
            echo=False
        )

        raw_output = output['choices'][0]['text'].strip()
        print(f"🔍 Raw LLM Output: '{raw_output}'")

        # Extract 0.xxx pattern
        match = re.search(r"0\.\d{3}", raw_output)
        if match:
            score = float(match.group(0))
            print(f"🎯 Parsed Score: {score:.3f}")
            return score

        # Fallback: extract any float between 0 and 1
        numbers = re.findall(r"0?\.\d+|\d+\.\d+", raw_output)
        for num_str in numbers:
            try:
                score = float(num_str)
                if 0.0 <= score <= 1.0:
                    clamped_score = round(score, 3)
                    print(f"🎯 Fallback Score: {clamped_score:.3f}")
                    return clamped_score
            except:
                continue

        print(f"⚠️ Could not parse score from: '{raw_output}'")
        return 0.500

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 0.500


# ============= MAIN =============
df = pd.read_csv(CSV_PATH)
print(f"📊 Loaded {len(df)} tasks.")
df['llm_priority_score'] = 0.0

for idx, row in df.iterrows():
    print(f"\n🚦 Processing Task {idx+1}/{len(df)} (ID: {row['task_id']})...")
    
    prompt = build_prompt(row)
    score = get_priority_score(prompt)
    
    df.at[idx, 'llm_priority_score'] = score
    print(f"✅ Final Priority Score: {score:.3f}")

    time.sleep(0.3)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n🎉 Scores saved to '{OUTPUT_PATH}'")
print("✅ Done! Scores should now be unique per task.")