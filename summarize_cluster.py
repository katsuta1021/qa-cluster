import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# === 0. 準備 ===
import transformers
transformers.logging.set_verbosity_error()

# sys.path にプロジェクトルートを追加
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.data_loader import load_questions_df

project_dir = Path("/workspace/ELYZA_test")
output_path = project_dir / "clustered_questions_all_patterns.xlsx"
project_dir.mkdir(exist_ok=True, parents=True)

df = load_questions_df()
df = df.dropna(subset=["question"]).copy()
questions = df["question"].tolist()

# === 1. E5 エンベディング用モデル ===
embedder = SentenceTransformer("intfloat/e5-large-v2", device="cuda")

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

# === 2. 要約プロンプト定義 ===
elyza_prompt = """あなたは日本語の質問文をクラスタリングしやすいように要約・再構成するアシスタントです。
以下の質問文を150文字以内で要点のみに変換してください。

【変換条件】
- あいさつや冗長な前置きは削除
- 曖昧な単語（例：「学習方法」など）は文脈から具体化
- ページ番号やスライド番号など文脈依存の参照は除去
- 内容語（専門用語）を中心に再構成
- クラスタリングで重みが付きやすいように意味的に濃縮

【質問文】
{text}

【変換後】
"""

elyza_generator = pipeline(
    "text-generation",
    model="elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
    device_map="auto",
    do_sample=False,
    max_new_tokens=60
)

def summarize_elyza_prompt(text):
    full_prompt = elyza_prompt.format(text=text)
    try:
        output = elyza_generator(full_prompt)[0]["generated_text"]
        return output.replace(full_prompt, "").strip()
    except Exception as e:
        print(f"[ELYZA要約失敗] {e}")
        return ""

# === 3. 要約 + クラスタリング ===
summary_funcs = {
    "elyza_prompt": summarize_elyza_prompt
}

for name, func in summary_funcs.items():
    print(f"=== {name} 要約開始 ===")
    start = time.time()
    summaries = [func(q) for q in tqdm(questions, desc=f"{name}: summarizing")]
    elapsed = time.time() - start
    print(f"★ {name}: {elapsed:.2f}s total, {elapsed/len(questions):.2f}s avg")

    df[f"{name}_summary"] = summaries

    print(f"=== {name} 埋め込み & クラスタリング ===")
    emb = embed_texts(summaries)
    labels, best_k, best_s = cluster_embeddings(emb)

    df[f"{name}_cluster"] = labels
    print(f"★ {name}: K={best_k}, silhouette={best_s:.3f}")

# === 4. 出力 ===
df.to_excel(output_path, index=False)
print(f"✅ 出力完了 → {output_path}")
