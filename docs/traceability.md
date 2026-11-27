# トレーサビリティマトリクス

| 要求 | 実装 | 確認 |
|---|---|---|
| R1、R2 | adapters、core、data | ゲート・分類・JSON 境界 |
| R3 | execute、quality.json | 障害集計、リトライ試験 |
| R4、R5 | compare、run、CLI | 回帰、不正比較、安全性拒否、CI 終了コード |
| R6 | save、read_runs、csv_payload | SQLite と JSON の往復、CSV 出力 |
| R7 | dashboard.py | 共通読み取り・出力関数の pytest、画面の手動試験 |
| R8 | .github/workflows/evaluation.yml、uv.lock | 同等 CLI のローカル実行 |
| R9 | Live、CLI の --allow-live | モック例外の秘匿テスト、実接続は未検証 |
