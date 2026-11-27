# LLM Evaluation Platform

複数モデルの品質を固定データセットで評価し、モデルやプロンプトの変更による回帰を検出するポートフォリオです。標準経路は API キー不要で、評価結果を JSON、CSV、任意の SQLite へ出力します。

## English overview

An offline-first LLM evaluation platform with deterministic mock baselines, configurable quality gates, regression detection, JSON/CSV artifacts, optional SQLite persistence, and a Streamlit dashboard. Optional provider adapters and a semantic judge require explicit live opt-in.

## 再現手順

ダッシュボードを含む標準検証は、リポジトリ直下で CI と同じ Linux x86_64（glibc 2.28 以上）、Python 3.14、uv を使用します。依存パッケージがキャッシュに揃っていれば同期・インストールに `--offline` を追加できます。

```bash
uv sync --locked
uv pip install --only-binary :all: --require-hashes -r requirements-dashboard.txt
uv run --no-sync pytest -q
uv run --no-sync python -m llm_eval run --output reports/good --db evaluation.db
uv run --no-sync python -m llm_eval run --model mock-bad --output reports/bad --db evaluation.db
uv run --no-sync python -m llm_eval compare reports/good/evaluation.json reports/bad/evaluation.json --output reports/regression.json
uv run --no-sync python -m llm_eval detector reports/good/evaluation.json reports/bad/evaluation.json --output reports/detector.json
```

`mock-bad` とこの比較は意図的に終了コード 1 を返すため、上記は一行ずつ実行してください。終了コード 0 は合格、1 は品質基準未達、2 は入力・設定エラーです。開発時は `--dataset data/dev.json` を指定し、最終確認には既定の `data/holdout.json` を使います。

## ダッシュボード

```bash
uv run --no-sync streamlit run dashboard.py
```

配布済み JSON を既定で表示します。サイドバーで JSON の追加読み込みや SQLite への切り替えができます。リモートの Streamlit は SSH ポート転送や既存の認証付き経路から利用してください。モードを切り替えると、実行時に記録されたモック評価と実LLM評価を確認できます。履歴が1件でも点を描画します。率の「未測定」と観測結果ゼロを区別します。回帰比較は同じ評価器版・データ・ポリシーで取得した2件を使用してください。

`uv.lock` は CLI と pytest の標準依存に使います。画面の直接依存は `requirements-dashboard.in` に記述し、推移依存を含めて `requirements-dashboard.txt` にバージョンとハッシュを固定しています。画面依存を変更したら、CI と同じ対象環境で次のコマンドを実行します。 `uv sync` を再実行した場合は、画面依存も再インストールしてください。

```bash
uv pip compile --generate-hashes --only-binary :all: --python-version 3.14 \
  --python-platform x86_64-manylinux_2_28 requirements-dashboard.in -o requirements-dashboard.txt
```

## 実モデルと意味的評価

接続先の利用とデータ送信を明示的に許可された環境で、クラウド SDK を `UV_HTTP_RETRIES=0 uv pip install -r requirements-providers.txt` で追加できます。Ollama は Python 標準ライブラリを使い、localhost:11434 で既に提供されているモデルを参照します。

- `run --model ollama:<ID> --allow-live --output reports/live`
- `openai:<ID>`、`anthropic:<ID>`、`gemini:<ID>` を指定可能。認証はそれぞれ `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`。
- `--judge <provider>:<ID>` は候補応答を判定モデルに送るため、送信先を含めて事前承認が必要。判定結果は独立した参考値として記録します。
- `--model-version` と `--prompt-version` を更新してください。`--prices INPUT OUTPUT` は 100 万トークンあたりの USD 単価です。
- 実モデル接続は未検証です。評価データとモデル識別ラベルには機密情報を含めないでください。レポートにプロンプト、応答、例外メッセージの原文は保存しません。

## 設計資料

- [学習用技術解説書](docs/learning-guide.md)
- [要求仕様書](docs/requirements.md)
- [テスト戦略書](docs/test-strategy.md)
- [テスト設計書](docs/test-design.md)
- [評価指標定義書](docs/metrics.md)
- [トレーサビリティマトリクス](docs/traceability.md)
- [品質レビュー記録](docs/review.md)
- [障害調査手順書](docs/runbook.md)

共通プロンプトを変更して比較する場合は候補実行に `--prompt-prefix-file <UTF-8 ファイル> --prompt-version 2` を追加します。共通プロンプトの本文はレポートに含まれません。モックはケース別の固定応答を返すため、プロンプトへの実際の反応は実モデルで計測します。
