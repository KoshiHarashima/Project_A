# Cursor AI Assistant Instructions

## Important Instructions

ユーザーから指示を受けた際は、まずこのCursor.mdファイルを読んでください。
このファイルには、プロジェクトに関する重要な情報と指示が含まれています。

## Notes

- ユーザーから指示を受けた際は、作業を開始する前にまずこのファイルを確認してください
- 適切な対応を取る前に、プロジェクト構造と要件を理解してください

## File Structure

- `regretNet/`
  - `base/`
    - `base_generator.py`
    - `base_generator_ca.py`
    - `base_net.py`
  - `baseline/`
    - `baseline.py`
  - `cfgs/`
    - `additive_1x10_uniform_config.py`
    - `additive_1x2_beta_11_config.py`
    - `additive_1x2_gamma_11_config.py`
    - `additive_1x2_uniform_04_03_config.py`
    - `additive_1x2_uniform_416_47_config.py`
    - `additive_1x2_uniform_config.py`
    - `additive_1x2_uniform_triangle_config.py`
    - `additive_1x3_constrained_config.py`
    - `additive_2x2_uniform_config.py`
    - `additive_2x3_uniform_config.py`
    - `additive_3x10_uniform_config.py`
    - `additive_5x10_uniform_config.py`
    - `CA_asym_uniform_12_15_config.py`
    - `CA_sym_uniform_12_config.py`
    - `unit_1x2_uniform_23_config.py`
    - `unit_1x2_uniform_config.py`
    - `unit_2x2_uniform_config.py`
  - `clip_ops/`
    - `clip_ops.py`
  - `data/`
    - `beta_11_generator.py`
    - `CA_asym_uniform_12_15_generator.py`
    - `CA_sym_uniform_12_generator.py`
    - `constrained_3item_generator.py`
    - `gamma_11_generator.py`
    - `uniform_01_generator.py`
    - `uniform_04_03_generator.py`
    - `uniform_12_generator.py`
    - `uniform_23_generator.py`
    - `uniform_416_47_generator.py`
    - `uniform_triangle_01_generator.py`
  - `experiments/`
  - `nets/`
    - `additive_net.py`
    - `ca2x2_net.py`
    - `constrained_additive_net.py`
    - `unit_net.py`
  - `plots/`
  - `trainer/`
    - `ca12_2x2.py`
    - `constrained_trainer.py`
    - `trainer.py`
  - `run_baseline.py`
  - `run_test.py`
  - `run_train.py`
  - `visualize_additive_1x2_beta_11.ipynb`
  - `visualize_additive_1x2_gamma_11.ipynb`
  - `visualize_additive_1x2_uniform_04_03.ipynb`
  - `visualize_additive_1x2_uniform_IC.ipynb`
  - `visualize_additive_1x2_uniform_triangle.ipynb`
  - `visualize_additive_1x2_uniform.ipynb`
  - `visualize_additive_1x3_constrained.ipynb`
  - `visualize_asymetric_daskalakis.ipynb`
  - `visualize_unit_1x2_uniform_23.ipynb`
  - `visualize_unit_1x2_uniform.ipynb`
- `rochetNet/`
  - `base/`
    - `base_generator.py`
  - `baseline/`
    - `baseline.py`
  - `cfgs/`
    - `additive_1x10_uniform_config.py`
    - `additive_1x2_uniform_04_03_config.py`
    - `additive_1x2_uniform_416_47_config.py`
    - `additive_1x2_uniform_config.py`
    - `additive_1x2_uniform_triangle_config.py`
    - `unit_1x2_uniform_23_config.py`
    - `unit_1x2_uniform_config.py`
  - `data/`
    - `uniform_01_generator.py`
    - `uniform_04_03_generator.py`
    - `uniform_12_generator.py`
    - `uniform_23_generator.py`
    - `uniform_416_47_generator.py`
    - `uniform_triangle_01_generator.py`
  - `experiments/`
  - `nets/`
    - `additive_net.py`
    - `unit_net.py`
  - `plots/`
  - `trainer/`
    - `trainer.py`
  - `run_baseline.py`
  - `run_test.py`
  - `run_train.py`
  - `debug.ipynb`
  - `visualize_additive_1x2_uniform_04_03.ipynb`
  - `visualize_additive_1x2_uniform_416_47.ipynb`
  - `visualize_additive_1x2_uniform_triangle.ipynb`
  - `visualize_additive_1x2_uniform.ipynb`
  - `visualize_unit_1x2_uniform_23.ipynb`
  - `visualize_unit_1x2_uniform.ipynb`
- `myersonNet/`
  - `baseline/`
    - `baseline.py`
  - `data/`
    - `generatedata.py`
  - `experiments/`
  - `nets/`
    - `net.py`
  - `plots/`
  - `utils/`
    - `cfg.py`
    - `plot.py`
  - `main.py`
  - `myerson.sh`
- `requirements.txt`
- `README.md`
- `LICENSE`
- `venv/`

## 環境概要

- 目的: 深層学習トレーニング
- 接続: Cursor → Remote SSH → AWS g4.xlarge (Amazon Linux 2)
- Python: 3.12.8
- フレームワーク: PyTorch
- GPUドライバ: 事前インストール済み

## プロジェクト概要

- **プロジェクト名**: Optimal Auctions through Deep Learning
- **実装内容**: RegretNet, RochetNet, MyersonNet
- **主要言語**: Python 3.12.8
- **フレームワーク**: PyTorch 2.9.0

## 環境設定

### 仮想環境の有効化
```bash
source venv/bin/activate
```

### 依存関係のインストール
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU確認
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## よく使うコマンド

### RegretNet
```bash
cd regretNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

### 主要な設定名
- `additive_1x2_uniform`
- `additive_1x2_gamma_11`
- `additive_1x3_constrained`

## 重要なファイルパス

### 設定ファイル
- `regretNet/cfgs/` - 各設定の設定ファイル
- `regretNet/data/` - データジェネレーター
- `regretNet/nets/` - ニューラルネットワーク定義
- `regretNet/trainer/` - トレーナークラス

### 実験結果
- `regretNet/experiments/[setting_name]/` - 実験結果とモデルファイル

## 注意事項・既知の問題

### テスト実行時の注意
- `run_test.py` は非常に時間がかかる場合があります
  - `num_misreports = 1000`、`gd_iter = 2000` などの設定により、1バッチあたり約6分かかる
  - 100バッチの場合、約10時間かかる可能性がある

### GPU対応
- 現在のコードは環境概要で示されているように, GPUで動作する. 

## 開発メモ
- 各プロジェクトごとに .venv を作成
- requirements.txt を最新に保つ
- 環境共有時はこの手順をチーム全員で共通利用
- インスタンスは毎回同一設定を使用



