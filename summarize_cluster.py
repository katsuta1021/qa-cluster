import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# === 0. 準備 ===
project_dir = Path("/workspace/ELYZA_test")
output_dir = project_dir / "result"
output_dir.mkdir(exist_ok=True, parents=True)
output_path = output_dir / "rinna_summary_clusters.xlsx"

# データ読み込み
import sys
sys.path.append(str(project_dir.parent))
from common.data_loader import load_questions_df
df = load_questions_df()
df = df.dropna(subset=["question"]).copy()
questions = df["question"].tolist()

# === 1. ELYZA モデル準備 ===
model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === 2. E5 エンベディングモデル ===
embedder = SentenceTransformer("intfloat/e5-large-v2", device=device)

def embed_texts(texts):
    return embedder.encode(
        [f"query: {s}" for s in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

def cluster_embeddings(embeddings):
    best_k, best_score = None, -1
    for k in range(8, 25, 2):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(embeddings)
        score = silhouette_score(embeddings, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42).fit(embeddings)
    return kmeans.labels_, best_k, best_score

# === 3. ELYZAによる要約関数 ===
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは講義質問をクラスタリングしやすい形式に整える NLP エキスパートです。"

def summarize_with_elyza(text: str) -> str:
    prompt_content = f"[依頼]これは大規模言語モデル（LLM）講座の質問です。以下の質問文を、：{text}[回答]"
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=prompt_content,
        e_inst=E_INST,
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=80,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return result.strip()

# === 4. 実行 ===
print("=== ELYZAによる要約開始 ===")
start = time.time()
summaries = [summarize_with_elyza(q) for q in tqdm(questions, desc="Summarizing")]
elapsed = time.time() - start
print(f"✅ 推論時間: 合計 {elapsed:.2f}s、平均 {elapsed/len(questions):.2f}s")

df["elyza_summary"] = summaries

# === 5. クラスタリング ===
print("=== クラスタリング開始 ===")
emb = embed_texts(summaries)
labels, best_k, best_s = cluster_embeddings(emb)
df["elyza_cluster"] = labels
print(f"✅ クラスタ数: K={best_k}, silhouette={best_s:.3f}")

# === 6. 保存 ===
df.to_excel(output_path, index=False)
print(f"📄 出力完了: {output_path}")
