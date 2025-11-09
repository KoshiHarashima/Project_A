# Optimal Auctions through Deep Learning
Implementation of "Optimal Auctions through Deep Learning" (https://arxiv.org/pdf/1706.03459.pdf)

## Getting Started

### Prerequisites
- Python 3.8+ (tested with Python 3.12.8)
- PyTorch (>=2.1.0, tested with PyTorch 2.9.0)
- Numpy and Matplotlib packages
- Easydict - `pip install easydict`

### Installation

#### Using Virtual Environment
```bash
# Create and activate virtual environment
source venv/bin/activate  # or: python -m venv venv

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Direct Installation
Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Running the experiments

### RegretNet

#### For Gradient-Based approach:
Default hyperparameters are specified in regretNet/cfgs/.  

#### For Sample-Based approach:
Modify the following hyperparameters in the config file specified in regretNet/cfgs/.
```
cfg.train.gd_iter = 0
cfg.train.num_misreports = 100
cfg.val.num_misreports = 100 # Number of val-misreports is always equal to the number of train-misreports
```

For training the network, testing the mechanism learnt and computing the baselines, run:
```bash
cd regretNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

#### Resuming Training from Checkpoint
To resume training from a saved checkpoint, modify `restore_iter` in the config file:
```python
__C.train.restore_iter = 180000  # Resume from iteration 180000
```
Then run training as usual. Note: This requires `save_data = True` in the config file to load training data files (`X.npy` and `ADV_*.npy`).

#### Data Saving for Restore
Set `save_data = True` in the config file to save training data, enabling checkpoint resumption:
```python
__C.save_data = True  # Saves X.npy and ADV_*.npy files
```

setting\_no  |      setting\_name |
 :---:   | :---: |
  (a)    |  additive\_1x2\_uniform |
  (b)   | unit\_1x2\_uniform\_23 |
  (c\)  | additive\_2x2\_uniform |
  (d)   | CA\_sym\_uniform\_12 |
  (e)    | CA\_asym\_uniform\_12\_15 |
  (f)   | additive\_3x10\_uniform |
  (g)  | additive\_5x10\_uniform |
  (h) |   additive\_1x2\_uniform\_416\_47
  (i) |   additive\_1x2\_uniform\_triangle
  (j) |   unit\_1x2\_uniform
  (k) |  additive\_1x10\_uniform
  (l) |   additive\_1x2\_uniform\_04\_03
  (m) |   unit\_2x2\_uniform
  (n) |   additive\_1x2\_beta\_11
  (o) |   additive\_1x2\_gamma\_11
  (p) |   additive\_1x2\_gamma\_21
  (q) |   additive\_1x2\_gamma\_31
  (r) |   additive\_1x3\_constrained


### RochetNet (Single Bidder Auctions)

Default hyperparameters are specified in rochetNet/cfgs/.  
For training the network, testing the mechanism learnt and computing the baselines, run:
```bash
cd rochetNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  additive\_1x2\_uniform |
  (b)   |   additive\_1x2\_uniform\_416\_47
  \(c\) |   additive\_1x2\_uniform\_triangle
  (d)   |   additive\_1x2\_uniform\_04\_03
  (e)   |  additive\_1x10\_uniform
  (f)   |   unit\_1x2\_uniform
  (g)   |   unit\_1x2\_uniform\_23
  
### MyersonNet (Single Item Auctions)
  
Default hyperparameters are specified in utils/cfg.py.  
For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd myersonNet
python main.py -distr [setting_name] or
bash myerson.sh
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  exponential 
  (b)   |   uniform
  \(c\) |   asymmetric\_uniform 
  (d)   |   irregular

 
## Settings

### Single Bidder
- **additive\_1x2\_uniform**: A single bidder with additive valuations over two items, where the items is drawn from U\[0, 1\].

- **unit\_1x2\_uniform\_23**: A single bidder with unit-demand valuations over two items, where the item values are drawn from U\[2, 3\].

- **additive\_1x2\_uniform\_416\_47**: Single additive bidder with preferences over two non-identically distributed items, where v<sub>1</sub> ∼ U\[4, 16\]and v<sub>2</sub> ∼ U\[4, 7\].

- **additive\_1x2\_uniform\_triangle**: A single additive bidder with preferences over two items, where (v<sub>1</sub>, v<sub>2</sub>) are drawn jointly and uniformly from a unit-triangle with vertices (0, 0), (0, 1) and (1, 0).

- **unit\_1x2\_uniform**: A single unit-demand bidder with preferences over two items, where the item values from U\[0, 1\]

- **additive\_1x2\_uniform\_04\_03**: A Single additive bidder with preferences over two items, where the item values v<sub>1</sub> ∼ U\[0, 4], v<sub>2</sub> ∼ U\[0, 3]

- **additive\_1x10\_uniform**: A single additive bidder and 10 items, where bidders draw their value for each item from U\[0, 1\].

- **additive\_1x2\_beta\_11**: A single additive bidder with preferences over two items, where the item values are drawn from Beta(1, 1), which is equivalent to U\[0, 1\].

- **additive\_1x2\_gamma\_11**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=1, θ=1) (mean=1, variance=1).

- **additive\_1x2\_gamma\_21**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=2, θ=1) (mean=2, variance=2).

- **additive\_1x2\_gamma\_31**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=3, θ=1) (mean=3, variance=3).

### Multiple Bidders
- **additive\_2x2\_uniform**: Two additive bidders and two items, where bidders draw their value for each item from U\[0, 1\]. 

- **unit\_2x2\_uniform**: Two unit-demand bidders and two items, where the bidders draw their value for each item from identical U\[0, 1\].

- **additive\_2x3\_uniform**: Two additive bidders and three items, where bidders draw their value for each item from U\[0, 1\]. 

- **CA\_sym\_uniform\_12**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub>, v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 2\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1\].

- **CA\_asym\_uniform\_12\_15**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub> ∼ U\[1, 2\], v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 5\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1].

- **additive\_3x10\_uniform**: 3 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].

- **additive\_5x10\_uniform**: 5 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].

### Constrained Allocation (制約付き配分)
- **additive\_1x3\_constrained**: 
単一のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, c\]（cはパラメータ、デフォルトは1.0）に従う。財3の配分確率には以下の制約が課される：
  - 下界制約: 財3の配分確率 ≥ max(0, 財1の配分確率 + 財2の配分確率-1)
  - 上界制約: 財3の配分確率 ≤ min(財1の配分確率, 財2の配分確率)
  この設定では、RegretNetに加えて配分確率の制約違反に対する罰則項が追加される。既存のregret罰則項と同様に、Lagrange乗数法を用いて制約違反を最小化する。実装は既存クラスを変更せず、新規クラス（`constrained_additive_net.py`、`constrained_trainer.py`など）として追加されている。

## Visualization

Allocation Probabilty plots for **unit\_1x2\_uniform_23** setting learnt by **regretNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc2.png" width="300">

Allocation Probabilty plots for **additive\_1x2\_uniform\_416\_47** setting learnt by **rochetNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc2.png" width="300">

For other allocation probability plots, check-out the ipython notebooks in `regretNet` or `rochetNet` folder.
