from llama_cpp import Llama
import pandas as pd
import re
import time
import os

# ============= CONFIG =============
MODEL_PATH = "./Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # ✅ Updated model name
CSV_PATH = "./50_synthetic_iov_tasks_final.csv"
OUTPUT_PATH = "./iov_tasks_with_priority_scores_llama3.csv"  # Different output file

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {CSV_PATH}")

# ============= LOAD MODEL (OPTIMIZED FOR 8GB RAM) =============
print("🧠 Loading Meta-Llama-3-8B-Instruct (GGUF, CPU mode)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Keep context reasonable for 8GB RAM
    n_threads=4,       # Ryzen 5 handles 4 threads well
    n_gpu_layers=0,    # CPU only
    n_batch=256,       # Smaller batch for low RAM
    verbose=False
)
print("✅ Llama-3 model loaded successfully.")


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

    # ✅ REVISED FOR LLAMA-3: Use official Llama-3 Instruct format
    # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI priority engine for Internet of Vehicles. Your job is to assign a priority score from 0.000 to 1.000 based on task urgency.
- Safety tasks at high speed with tight deadlines = score near 1.0
- Infotainment at low speed with loose deadlines = score near 0.1
- Medium urgency = score around 0.5
Output ONLY the score as a number like 0.753 — nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>

Task Type: Safety
Criticality: High
Deadline: 50 ms
Vehicle Speed: 98 km/h
Context: Emergency braking needed at high speed.
Score:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

0.987<|eot_id|><|start_header_id|>user<|end_header_id|>

Task Type: Infotainment
Criticality: Low
Deadline: 860 ms
Vehicle Speed: 0 km/h
Context: Music streaming, vehicle stationary.
Score:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

0.125<|eot_id|><|start_header_id|>user<|end_header_id|>

Task Type: Diagnostics
Criticality: Medium
Deadline: 274 ms
Vehicle Speed: 87 km/h
Context: Predictive maintenance while moving.
Score:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

0.543<|eot_id|><|start_header_id|>user<|end_header_id|>

Task Type: {task_type}
Criticality: {criticality}
Deadline: {deadline_ms} ms
Vehicle Speed: {vehicle_speed} km/h
Context: {context_text}
Score:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def get_priority_score(prompt):
    try:
        output = llm(
            prompt,
            max_tokens=15,      # Give it room to generate the score
            temperature=0.3,    # Slight randomness for diversity
            top_p=0.95,
            echo=False
        )

        raw_output = output['choices'][0]['text'].strip()
        print(f"🔍 Raw LLM Output: '{raw_output}'")  # DEBUG: See what model actually says

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
        return 0.500  # Last resort fallback

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 0.500


# ============= MAIN EXECUTION =============
df = pd.read_csv(CSV_PATH)
print(f"📊 Loaded {len(df)} tasks.")
df['llm_priority_score'] = 0.0

for idx, row in df.iterrows():
    print(f"\n🚦 Processing Task {idx+1}/{len(df)} (ID: {row['task_id']})...")
    
    prompt = build_prompt(row)
    score = get_priority_score(prompt)
    
    df.at[idx, 'llm_priority_score'] = score
    print(f"✅ Final Priority Score: {score:.3f}")

    time.sleep(0.4)  # Prevent CPU/RAM overload

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n🎉 Priority scores saved to '{OUTPUT_PATH}'")
print("✅ Done! Scores generated using Meta-Llama-3-8B-Instruct.")