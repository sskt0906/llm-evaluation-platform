# 障害調査手順書

1. CI の評価成果物を取得し、`evaluation.json` の run ID、実行日時、版、データセットハッシュを照合する。
2. `gate.reasons` と不合格ケースの ID、分類、試行回数を確認する。障害注入ケースは `expected_error` と終端エラーが一致していることを確認する。
3. 実接続の 429、500、timeout は提供側の利用制限、稼働状況、接続状態を調べる。例外原文や認証情報は共有レポートに転記しない。
4. `compare` で候補と基準を照合する。終了コード 2 の場合はデータセット・設定・評価器版・モード・入力パスを確認する。
5. ネットワーク障害時は既存キャッシュで `uv sync --locked --offline` と `uv pip install --offline --only-binary :all: --require-hashes -r requirements-dashboard.txt` を試す。画面用依存がなければ生成済み JSON/CSV または SQLite を共有する。
