# 評価指標定義書

| 指標 | 定義・分母 |
|---|---|
| Pass Rate | 全チェックに合格した応答数 / 実行対象全件 |
| Accuracy | 正解を返した件数 / 事実性チェック対象件数 |
| Faithfulness | 応答内の全行が根拠集合に含まれる件数 / 根拠付きケース件数 |
| Hallucination Rate | 未支持行を検出した回答数 / 根拠付きケース件数。実行失敗は Error Rate と評価不能件数に記録 |
| Instruction Following Rate | 指示に合致した件数 / 指示評価対象件数 |
| Schema Compliance Rate | 許可されたキーと整数型・値域を満たす件数 / スキーマ対象件数 |
| Abstention Accuracy | 正しく回答保留した件数 / 情報不足ケース件数 |
| Error Rate | 終端エラー件数 / 実行対象全件 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2TP / (2TP + FP + FN) |
| False Positive Rate | FP / (FP + TN) |
| False Negative Rate | FN / (FN + TP) |
| P50 Latency / P95 Latency | 実測ミリ秒の nearest-rank 50・95 パーセンタイル。再試行と待機を含む |
| Token Usage | 入出力合計。全件が揃わない場合 null とし、観測分と測定件数を併記 |
| Cost per Successful Task | 全ケースの観測可能な入出力費用合計 / 合格件数。いずれかのケースの費用が不明なら null |

分母ゼロと未測定は JSON の null、CSV では該当セルを空欄とする。正例は人手で定義した未支持の回答。分類不能は混同行列に別集計し、陽性の分類不能は再現率・F1・偽陰性率で FN に加算する。検出器の合否は正常と劣化のラベルを合算して判定する。モックのトークンは空白区切りの疑似値で、障害ケースには記録がない。実モデルでは SDK が返すトークン件数を使う。失敗試行や意味判定の課金は推定対象外で、請求額との照合には提供元の明細が必要。

フォールト注入はモック専用で、3 件とも Pass Rate の失敗に含める。現行データでの正常モデルの上限は 6/9。実モデルではフォールト注入 3 件を対象外とし、呼び出し後に生じた障害は全件集計する。
