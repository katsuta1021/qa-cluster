import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import T5Tokenizer, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from sentence_transformers import SentenceTransformer

# sys.path にプロジェクトルートを追加
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.data_loader import load_questions_df

# === 0. 準備 ===
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

# === 2. 要約パターン定義 ===

# 2-1: text-generation モード (ELYZA)
elyza_prompt = """以下の自然言語による質問文を、クラスタリングや分類がしやすいように次の条件で変換してください：

- あいさつや冗長な前置きは削除
- 曖昧な単語（例：「学習方法」など）は文脈から具体化
- ページ番号やスライド番号など文脈依存の参照は除去
- 内容語（専門用語）を中心に再構成
- クラスタリングで重みが付きやすいように意味的に濃縮

元の質問文：{text}
分類に適した質問文：
"""

elyza_generator = pipeline(
    "text-generation",
    model="elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
    device_map="auto",
    do_sample=False,
    max_new_tokens=60  # 生成トークン数だけ制限
)
def summarize_elyza_prompt(text):
    full_prompt = elyza_prompt.format(text=text)
    output = elyza_generator(full_prompt)[0]["generated_text"]
    return output.replace(full_prompt, "").strip()

# 2-2: summarization モード（T5）

tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", use_fast=False)

t5_summarizer = pipeline(
    "summarization",
    model="sonoisa/t5-base-japanese",
    tokenizer=tokenizer
)

def summarize_t5_summarization(text):
    try:
        return t5_summarizer(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"].strip()
    except:
        return ""

# 2-3: text2text モード（T5）
t5_tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("sonoisa/t5-base-japanese")

def summarize_t5_text2text(text):
    input_ids = t5_tokenizer.encode(f"要約: {text}", return_tensors="pt")
    output_ids = t5_model.generate(input_ids, max_length=60, min_length=20, do_sample=False)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === 3. 各パターン実行 ===
summary_funcs = {
    "elyza_prompt": summarize_elyza_prompt,
    "t5_summarization": summarize_t5_summarization,
    "t5_text2text": summarize_t5_text2text
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
print(f"✅ 全パターン出力完了 → {output_path}")

