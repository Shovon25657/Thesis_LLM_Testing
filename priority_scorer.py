from llama_cpp import Llama
import pandas as pd
import re
import time
import os

# ============= CONFIG =============
MODEL_PATH = "./Phi-3-mini-4k-instruct.Q4_0.gguf"
CSV_PATH = "./50_synthetic_iov_tasks_final.csv"  # ✅ Your file — already correct!
OUTPUT_PATH = "./iov_tasks_with_priority_scores.csv"

# Check files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {CSV_PATH}")

# ============= LOAD MODEL (LIGHTWEIGHT) =============
print("🧠 Loading Phi-3-mini-4k-instruct (GGUF, CPU mode, optimized for 8GB RAM)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Reduced context to save RAM
    n_threads=4,       # Use 4 threads (Ryzen 5 handles this well)
    n_gpu_layers=0,    # CPU only
    n_batch=512,       # Reduce batch size for low RAM
    verbose=False      # No debug spam
)
print("✅ Model loaded successfully.")


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

    # Few-shot examples to guide scoring
    examples = """
Example 1:
Task Type: Safety
Criticality: High
Data Size: 162 KB
Deadline: 50 ms
Vehicle Speed: 91 km/h
Location: Urban
Network: Congested
Distance to Edge: 1.13 km
Computation: 3.58e+09 FLOPs
Cost Preference: Cost-Sensitive
Time of Day: Peak
Mobility Pattern: High-Mobility
Context: Emergency braking needed, high speed, low latency tolerance.
Priority Score: 0.987

Example 2:
Task Type: Infotainment
Criticality: Low
Data Size: 4021 KB
Deadline: 860 ms
Vehicle Speed: 0 km/h
Location: Urban
Network: Congested
Distance to Edge: 1.78 km
Computation: 1.25e+09 FLOPs
Cost Preference: Cost-Sensitive
Time of Day: Peak
Mobility Pattern: Stable
Context: Non-urgent infotainment, user cost-sensitive, vehicle stationary.
Priority Score: 0.125

Example 3:
Task Type: Diagnostics
Criticality: Medium
Data Size: 669 KB
Deadline: 274 ms
Vehicle Speed: 87 km/h
Location: Urban
Network: Good
Distance to Edge: 1.6 km
Computation: 6.45e+09 FLOPs
Cost Preference: Balanced
Time of Day: Peak
Mobility Pattern: High-Mobility
Context: Moderate urgency, moderate resource needs, moving vehicle.
Priority Score: 0.543
"""

    # Phi-3 Instruct format
    prompt = f"""<|system|>
You are an intelligent priority scoring engine for Internet of Vehicles (IoV) task offloading.
Given the task context and features, output a priority score between 0.000 and 1.000 (3 decimal places).
Higher score = higher priority (urgent, safety-critical, low-latency).
Lower score = lower priority (delay-tolerant, cost-sensitive).
Use the examples above to learn the scoring pattern.<|end|>
<|user|>
Now score this task:

Task Type: {task_type}
Criticality: {criticality}
Data Size: {data_size_kb} KB
Deadline: {deadline_ms} ms
Vehicle Speed: {vehicle_speed} km/h
Location: {location}
Network: {network}
Distance to Edge: {distance_km} km
Computation: {comp_flops} FLOPs
Cost Preference: {cost_pref}
Time of Day: {time_of_day}
Mobility Pattern: {mobility}
Context: {context_text}

Priority Score:<|end|>
<|assistant|>
"""
    return examples + prompt


def get_priority_score(prompt):
    try:
        output = llm(
            prompt,
            max_tokens=10,
            stop=["\n", "Task", "<|", " ", "."],  # Stop early
            temperature=0.1,   # Deterministic
            top_p=0.9,
            echo=False
        )

        text = output['choices'][0]['text'].strip()
        
        # Extract 0.xxx
        match = re.search(r"0\.\d{3}", text)
        if match:
            return float(match.group(0))
        
        # Fallback: any float
        numbers = re.findall(r"\d+\.\d+", text)
        if numbers:
            score = float(numbers[0])
            return round(max(0.0, min(1.0, score)), 3)
        
        print(f"⚠️ No score found. Raw: '{text}'")
        return 0.500

    except Exception as e:
        print(f"⚠️ Error generating score: {e}")
        return 0.500


# ============= MAIN EXECUTION =============
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"📊 Loaded {len(df)} tasks.")

df['llm_priority_score'] = 0.0

print("🚦 Starting priority scoring...")

for idx, row in df.iterrows():
    print(f"\nProcessing Task {idx+1}/{len(df)} (ID: {row['task_id']})...")
    
    prompt = build_prompt(row)
    score = get_priority_score(prompt)
    
    df.at[idx, 'llm_priority_score'] = score
    print(f"✅ Generated Priority Score: {score:.3f}")

    # Small delay to avoid CPU/RAM overload on 8GB system
    time.sleep(0.4)

# Save results
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n🎉 Priority scores saved to '{OUTPUT_PATH}'")
print("✅ Done! You can now use these scores in your thesis optimization model.")