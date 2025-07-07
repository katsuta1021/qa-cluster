# 質問クラスタリング用 正規化・分類スクリプト

このスクリプトは、大規模言語モデル（LLM）講座などで収集された受講者からの質問リストを整形・分類し、クラスタリングに適した形式へ変換するものです。

## 🎯 機能概要

- 受講者の質問文から、以下の処理を自動で行います：
  - **冗長な前置き（挨拶・謝辞）の除去**
  - **図・ページ番号の `<FIG>` 置換**
  - **専門語を保持した簡潔な正規化**
  - **クラスタリング向け埋め込みベクトルの生成**
  - **クラスタ ID の付与（HDBSCAN）**
  - **図表参照など文脈依存な質問のタグ付け**

## 🧠 使用モデル

- 正規化・分類には `ELYZA-japanese-Llama-2-7b-instruct` を使用
- ベクトル化には `intfloat/e5-large-v2` を使用（変更可能）

## 📂 入出力ファイル

| 種別 | 説明 |
|------|------|
| 入力 | `xlsx` ファイル（質問列が `"question"`） |
| 出力 | 元ファイル＋以下の列を追加した `xlsx`：  
  - `canonical_question`：正規化された質問文  
  - `tags`：トピックに基づくタグ  
  - `status`：`ready` / `needs_context`（図表参照など）  
  - `elyza_cluster`：自動クラスタ ID（-1 は分類外）

## 🛠️ 使用方法（例）

```bash
python summarize_cluster.py --input_file "input_questions.xlsx" --output_file "clustered_questions.xlsx"
```

## 🔖 ライブラリ要件（例）

```bash
pip install torch transformers hdbscan pandas openpyxl
```

## 📝 出力例

| question | canonical_question | tags | status | elyza_cluster |
|----------|--------------------|------|--------|----------------|
| この図についてですが… | <FIG> の注意点は | ref_figure | needs_context | -1 |
| バッチ正規化について詳しく | バッチ正規化の目的 | バッチ正規化 | ready | 3 |

## 🔧 カスタマイズ

- ベクトル埋め込みモデルやクラスタ条件（`min_cluster_size`）はスクリプト中で調整可能です。
- 図表参照のトークン（`<FIG>`）やタグルールも `normalize_question_with_elyza` 関数内で変更できます。

---

このツールを使えば、限られた講義時間内でも情報密度の高い質問対応・整理が可能です。
