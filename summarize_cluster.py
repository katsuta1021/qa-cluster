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
output_path = output_dir / "elyza_summary_clusters.xlsx"
lecture_title = "大規模言語モデル(LLM)講座"

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
# ---- 定数（ELYZA 系） 
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS,  E_SYS  = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = f"あなたは講義「{lecture_title}」での質問をクラスタリングしやすい形式に整える NLP エキスパートです。"
def normalize_question_with_elyza(text: str) -> dict:
    """
    元質問を正規化し、クラスタリング向け JSON を返す。
    返り値:
        {
            "canonical_question": str,   # 30 字以内
            "tags": [str, ...],          # 1〜3 語
            "status": "ready"|"needs_context"
        }
    """
    # ---- プロンプト本体 ----
    system_instructions = """
- 原文の日本語専門用語は必ず残す
- あいさつ・謝辞など回答に不要な語句は削除する
- ページ／図／スライド番号は全て <FIG> に置換する
- canonical_question は 30 字以内で意味核を残す
- <FIG> が含まれる場合 status を "needs_context" とし tags に "ref_figure" を追加
- それ以外は status を "ready"
- tags は主要トピック名詞を 1〜3 語、日本語で列挙
出力は **以下 JSON** のみ:
{"canonical_question": string,
 "tags": [string],
 "status": "ready" | "needs_context"}

 以下の条件を厳密に守ってください。
- 出力は必ず JSON フォーマットで返してください。
- JSON キーは "canonical_question", "tags", "status" の3つだけです。
- JSON 以外の文字は絶対に出力しないでください。
- canonical_question は30文字以内、日本語の意味核であること。
- tags は1〜3語の日本語トピック名詞（例: "勾配降下法", "学習率"）。
- 入力文に図表やページ指定が含まれる場合、status は "needs_context" とし、tags に "ref_figure" を追加。
- 上記以外は status を "ready" としてください。
    """.strip()

    prompt_content = f"[依頼]\n{text}\n[回答]"

    prompt = "{bos}{b_inst} {sys}{inst} {e_inst}".format(
        bos=tokenizer.bos_token,
        b_inst=B_INST,
        sys=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        inst=f"{system_instructions}\n\n{prompt_content}",
        e_inst=E_INST,
    )

    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=320,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw_text = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=False  # ← 念のため特別トークンも残す
    ).strip()

    result = {
        "_raw_text": raw_text  # ← デバッグ用にそのまま返す
    }

    import json, re
    json_str = re.search(r"\{.*\}", raw_text, re.S)
    if not json_str:
        print("[WARNING] JSON形式で出力されませんでした")
        return result

    try:
        result.update(json.loads(json_str.group(0)))
    except Exception as e:
        print("[ERROR] JSON decode failed:", e)
        print("内容:", json_str.group(0))
    return result

# === 4. 実行 ===
print("=== ELYZAによる要約開始 ===")
start = time.time()

raw_texts = [] 
norm_records = []
for q in tqdm(questions, desc="Normalizing"):
    result = normalize_question_with_elyza(q)
    norm_records.append(result)
    raw_texts.append(result.get("_raw_text", ""))  

df["canonical_question"] = [r.get("canonical_question", "") for r in norm_records]
df["tags"] = [", ".join(r.get("tags", [])) for r in norm_records]
df["status"] = [r.get("status", "") for r in norm_records]
df["elyza_raw"] = raw_texts

elapsed = time.time() - start
print(f"✅ 推論時間: 合計 {elapsed:.2f}s、平均 {elapsed/len(questions):.2f}s")

# df["elyza_summary"] = summaries

# === 5. クラスタリング ===
print("=== クラスタリング開始 ===")
canonical_questions = [r.get("canonical_question", "") for r in norm_records]
emb = embed_texts(canonical_questions)  
labels, best_k, best_s = cluster_embeddings(emb)
df["elyza_cluster"] = labels
print(f"✅ クラスタ数: K={best_k}, silhouette={best_s:.3f}")

# === 6. 保存 ===
df.to_excel(output_path, index=False)
print(f"📄 出力完了: {output_path}")
