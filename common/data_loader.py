# 質問フォームExcelから列名正規化済みのDataFrameを取得（question/answer/timestamp）

from pathlib import Path
import pandas as pd

def load_questions_df(xlsx_path: Path = Path("/workspace/data/【大規模言語モデル(LLM)講座2023】講義内容質問フォーム .xlsx")) -> pd.DataFrame:
    # Excel読み込み
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl", header=0)
    df.columns = [str(c).strip() for c in df.columns]

    # 列名リネーム
    rename_map = {}
    for col in df.columns:
        if "質問" in col:
            rename_map[col] = "question"
        elif "回答" in col:
            rename_map[col] = "answer"
        elif "タイムスタンプ" in col or "時刻" in col:
            rename_map[col] = "timestamp"
    df = df.rename(columns=rename_map)

    # タイムスタンプ修正（Excel日付のとき）
    if "timestamp" in df.columns and df["timestamp"].dtype in ["float64", "int64"]:
        df["timestamp"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["timestamp"], unit="D")

    # タイムスタンプを文字列に統一
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df
