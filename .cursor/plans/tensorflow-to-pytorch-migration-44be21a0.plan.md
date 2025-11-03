<!-- 44be21a0-d766-4fb7-aae8-675e8e34c968 0d23b43e-5c9a-49cd-be38-c7c9359db950 -->
# TensorFlow 1 → PyTorch マイグレーション計画（regretNet, rochetNet, myersonNet）

## 対象範囲

### regretNet

- `regretNet/nets/*` (additive_net.py, unit_net.py, ca2x2_net.py)
- `regretNet/trainer/trainer.py`, `regretNet/trainer/ca12_2x2.py`
- `regretNet/clip_ops/clip_ops.py`
- `regretNet/base/base_net.py`
- `regretNet/run_train.py`, `regretNet/run_test.py` (必要に応じて)

### rochetNet

- `rochetNet/nets/*` (additive_net.py, unit_net.py)
- `rochetNet/trainer/trainer.py`
- `rochetNet/base/base_net.py` (nets/additive_net.py内の関数として存在)
- `rochetNet/run_train.py`, `rochetNet/run_test.py` (必要に応じて)
- 注意: rochetNetにはclip_opsは存在しない

### myersonNet

- `myersonNet/nets/net.py` (MyersonNetクラス)
- `myersonNet/main.py` (必要に応じて)
- 注意: myersonNetは独自の構造で、trainerパターンを使用していない

## 作業手順

### Phase 1: 環境準備

1. **ブランチ作成**: `migrate/port-no-test` ブランチを作成
2. **requirements.txt作成**: 

- `torch>=2.1.0` (Dockerfileと整合性を取る)
- `numpy`, `matplotlib`, `easydict` 等の依存関係を固定
- ハッシュベースのロックファイルは後段階で作成

### Phase 2: 静的解析

3. **TF1トークン抽出（全ディレクトリ）**: 

- `tf.Session`, `tf.InteractiveSession` → PyTorch不要（インプレース実行）
- `tf.placeholder` → 関数引数
- `tf.get_variable`, `tf.variable_scope` → `torch.nn.Module`パラメータ
- `tf.train.*Optimizer` → `torch.optim.*`
- `tf.py_func` → 直接NumPy関数呼び出し

4. **NumPy APIチェック**: 非推奨型エイリアス（`np.int`等）を検出

### Phase 3: 移植実装（小さなコミット単位）

#### regretNet

5. **regretNet/base/base_net.py**: 

- `BaseNet`を`torch.nn.Module`に変更
- `create_var`を`torch.nn.Parameter`に置換
- 重み減衰は`torch.optim.Adam(weight_decay=...)`で処理

6. **regretNet/nets/additive_net.py**: 

- `Net`を`torch.nn.Module`継承に変更
- `build_net()`を`__init__`内の`torch.nn.Linear`に置換
- `inference()`メソッドは引数`x`をテンソルとして受け取り、PyTorch演算で実装

7. **regretNet/nets/unit_net.py**, **regretNet/nets/ca2x2_net.py**: 同様に移植
8. **regretNet/clip_ops/clip_ops.py**: 

- `tf.assign`を`torch.clamp`とin-place操作に置換
- `tf.py_func`は直接NumPy関数呼び出しに変更（PyTorchテンソル→NumPy→PyTorch変換）

9. **regretNet/trainer/trainer.py**: 

- セッション管理を削除（インプレース実行）
- プレースホルダーを関数引数に変更
- オプティマイザーを`torch.optim.*`に変更
- モデル保存を`torch.save()`に変更

10. **regretNet/trainer/ca12_2x2.py**: 同様に移植

#### rochetNet

11. **rochetNet/base/base_net.py** (または nets/additive_net.py内の関数): 

- `create_var`と`activation_summary`関数をPyTorchベースに移植
- `BaseNet`クラスを`torch.nn.Module`に変更（存在する場合）

12. **rochetNet/nets/additive_net.py**, **rochetNet/nets/unit_net.py**: 同様に移植
13. **rochetNet/trainer/trainer.py**: 

- regretNetのtrainer.pyと同様の移植（clip_op_lambdaパラメータは不要）
- セッション管理を削除、オプティマイザーを`torch.optim.*`に変更

#### myersonNet

14. **myersonNet/nets/net.py**: 

- `MyersonNet`クラスを`torch.nn.Module`継承に変更
- `tf.InteractiveSession`を削除
- `tf.placeholder`を関数引数に変更
- `tf.Variable`を`torch.nn.Parameter`に置換
- `nn_eval`, `nn_train`, `nn_test`メソッドをPyTorch演算に変更

15. **myersonNet/main.py**: 

- 必要に応じてマイグレーション後のAPIに合わせて調整

### Phase 4: 出力と検証

16. **各ファイルのパッチ作成**: `git diff`で各変更をパッチファイルとして保存
17. **軽微チェック**: 

- `flake8`, `isort`でimport解決と構文チェック（全ディレクトリ）
- Dockerイメージビルド（依存解決確認のみ、実行テストなし）

### Phase 5: ロックとタグ

18. **requirements.txt固定**: ハッシュベースのロックファイル作成
19. **Dockerイメージタグ固定**: `v0.1-migrated-no-test`タグ作成

### Phase 6: ドキュメント更新

20. **README.md更新**: 

- 「Getting Started」セクションのPythonバージョンとライブラリ情報を更新（TensorFlow → PyTorch）
- 該当部分のみ最小限の変更（他のセクションは保持）

## 重要な制約

- **公開API保持**: 
- regretNet: `Net.__init__(config)`, `Net.inference(x)`, `Trainer.__init__(config, mode, net, clip_op_lambda)`などのインターフェースは変更しない
- rochetNet: `Net.__init__(config)`, `Net.inference(x)`, `Trainer.__init__(config, mode, net)`などのインターフェースは変更しない
- myersonNet: `MyersonNet.__init__(args, train_data, test_data)`, `nn_train()`, `nn_test()`などのインターフェースは変更しない
- **設定ファイル互換**: `config`オブジェクトの構造と使用方法は変更しない
- **段階的コミット**: 各ファイルの変更は個別のコミットとして管理
- **README.md最小限変更**: ライブラリ関連の変更のみ対応部分を更新（「Getting Started」セクションの依存関係記述のみ）

## 主要な置換パターン

| TensorFlow 1 | PyTorch |
|--------------|---------|
| `tf.placeholder` | 関数引数 |
| `tf.get_variable` | `torch.nn.Parameter` |
| `tf.variable_scope` | `torch.nn.Module`の属性 |
| `tf.train.AdamOptimizer` | `torch.optim.Adam` |
| `tf.Session().run()` | 直接テンソル操作 |
| `tf.assign(x, y)` | `x.data = y` または `x.copy_(y)` |
| `tf.clip_by_value` | `torch.clamp` |
| `tf.py_func` | NumPy関数直接呼び出し |

### To-dos

- [ ] ブランチ migrate/port-no-test を作成
- [ ] requirements.txtを作成し、torch、numpy等の依存関係を固定
- [ ] 静的解析: 全ディレクトリ（regretNet, rochetNet, myersonNet）のTF1トークンを抽出し、置換箇所リストを作成
- [ ] 静的解析: 全ディレクトリの古いNumPy API（非推奨型エイリアス等）を抽出
- [ ] regretNet/base/base_net.pyをPyTorchベースに移植
- [ ] regretNet/nets/additive_net.pyをPyTorchベースに移植
- [ ] regretNet/nets/unit_net.pyをPyTorchベースに移植
- [ ] regretNet/nets/ca2x2_net.pyをPyTorchベースに移植
- [ ] regretNet/clip_ops/clip_ops.pyをPyTorchベースに移植
- [ ] regretNet/trainer/trainer.pyをPyTorchベースに移植
- [ ] regretNet/trainer/ca12_2x2.pyをPyTorchベースに移植
- [ ] rochetNet/base/base_net.py（またはnets内）をPyTorchベースに移植
- [ ] rochetNet/nets/additive_net.py, unit_net.pyをPyTorchベースに移植
- [ ] rochetNet/trainer/trainer.pyをPyTorchベースに移植
- [ ] myersonNet/nets/net.py（MyersonNetクラス）をPyTorchベースに移植
- [ ] myersonNet/main.pyを必要に応じて調整
- [ ] 各ファイルごとにパッチ（diff）を作成し、2行の説明を付す
- [ ] flake8/isortでimport解決と構文エラーをチェック（全ディレクトリ）
- [ ] Dockerイメージをビルドして依存解決を確認（実行テストなし）
- [ ] requirements.txtをハッシュベースでロックし、Dockerイメージタグv0.1-migrated-no-testを作成
- [ ] README.mdの「Getting Started」セクションのみを更新（Pythonバージョン、TensorFlow→PyTorch）