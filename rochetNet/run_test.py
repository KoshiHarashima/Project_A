from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np


from nets import *
from cfgs import *
from data import *
from trainer import *

print("Setting: %s"%(sys.argv[1]))
setting = sys.argv[1]


if setting == "additive_1x2_uniform":
    cfg = additive_1x2_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_416_47":
    cfg = additive_1x2_uniform_416_47_config.cfg
    Net = additive_net.Net
    Generator = uniform_416_47_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_04_03":
    cfg = additive_1x2_uniform_04_03_config.cfg
    Net = additive_net.Net
    Generator = uniform_04_03_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_triangle":
    cfg = additive_1x2_uniform_triangle_config.cfg
    Net = additive_net.Net
    Generator = uniform_triangle_01_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x10_uniform":
    cfg = additive_1x10_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x3_uniform_0_1":
    cfg = additive_1x3_uniform_0_1_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer

elif setting == "unit_1x2_uniform":
    cfg = unit_1x2_uniform_config.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer
     
elif setting == "unit_1x2_uniform_23":
    cfg = unit_1x2_uniform_23_config.cfg
    Net = unit_net.Net
    Generator = uniform_23_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_11":
    cfg = additive_1x2_gamma_11_config.cfg
    Net = additive_net.Net
    Generator = gamma_11_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x3_gamma_1_0_1_0":
    cfg = additive_1x3_gamma_1_0_1_0_config.cfg
    Net = additive_net.Net
    Generator = gamma_11_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_21":
    cfg = additive_1x2_gamma_21_config.cfg
    Net = additive_net.Net
    Generator = gamma_21_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_31":
    cfg = additive_1x2_gamma_31_config.cfg
    Net = additive_net.Net
    Generator = gamma_31_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_22":
    cfg = additive_1x2_gamma_22_config.cfg
    Net = additive_net.Net
    Generator = gamma_22_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_41":
    cfg = additive_1x2_gamma_41_config.cfg
    Net = additive_net.Net
    Generator = gamma_41_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_101":
    cfg = additive_1x2_gamma_101_config.cfg
    Net = additive_net.Net
    Generator = gamma_101_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_100_1":
    cfg = additive_1x2_gamma_100_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_100_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_20_1":
    cfg = additive_1x2_gamma_20_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_20_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_1":
    cfg = additive_1x2_gamma_0_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_2":
    cfg = additive_1x2_gamma_0_2_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_2_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_3":
    cfg = additive_1x2_gamma_0_3_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_3_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_4":
    cfg = additive_1x2_gamma_0_4_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_4_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_5":
    cfg = additive_1x2_gamma_0_5_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_5_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_6":
    cfg = additive_1x2_gamma_0_6_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_6_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_7":
    cfg = additive_1x2_gamma_0_7_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_7_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_8":
    cfg = additive_1x2_gamma_0_8_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_8_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_0_9":
    cfg = additive_1x2_gamma_0_9_config.cfg
    Net = additive_net.Net
    Generator = gamma_0_9_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_0":
    cfg = additive_1x2_gamma_1_0_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_0_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_1":
    cfg = additive_1x2_gamma_1_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_2":
    cfg = additive_1x2_gamma_1_2_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_2_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_23_1":
    cfg = additive_1x2_gamma_1_23_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_23_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_3":
    cfg = additive_1x2_gamma_1_3_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_3_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_4":
    cfg = additive_1x2_gamma_1_4_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_4_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_25_1":
    cfg = additive_1x2_gamma_1_25_1_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_25_1_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_5":
    cfg = additive_1x2_gamma_1_5_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_5_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_6":
    cfg = additive_1x2_gamma_1_6_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_6_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_7":
    cfg = additive_1x2_gamma_1_7_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_7_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_8":
    cfg = additive_1x2_gamma_1_8_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_8_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_1_9":
    cfg = additive_1x2_gamma_1_9_config.cfg
    Net = additive_net.Net
    Generator = gamma_1_9_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_gamma_2_0":
    cfg = additive_1x2_gamma_2_0_config.cfg
    Net = additive_net.Net
    Generator = gamma_2_0_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_beta_11":
    cfg = additive_1x2_beta_11_config.cfg
    Net = additive_net.Net
    Generator = beta_11_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_beta_12":
    cfg = additive_1x2_beta_12_config.cfg
    Net = additive_net.Net
    Generator = beta_12_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_beta_21":
    cfg = additive_1x2_beta_21_config.cfg
    Net = additive_net.Net
    Generator = beta_21_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_beta_13":
    cfg = additive_1x2_beta_13_config.cfg
    Net = additive_net.Net
    Generator = beta_13_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_beta_31":
    cfg = additive_1x2_beta_31_config.cfg
    Net = additive_net.Net
    Generator = beta_31_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x3_constrained_c1":
    cfg = additive_1x3_constrained_c1_config.cfg
    Net = constrained_additive_net.Net
    Generator = constrained_3item_generator.Generator
    Trainer = constrained_trainer.Trainer

elif setting == "additive_1x10_gamma_11":
    cfg = additive_1x10_gamma_11_config.cfg
    Net = additive_net.Net
    Generator = gamma_11_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_m05":
    cfg = additive_1x2_normal_corr_m05_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_m05_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_0":
    cfg = additive_1x2_normal_corr_0_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_0_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_05":
    cfg = additive_1x2_normal_corr_05_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_05_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_07":
    cfg = additive_1x2_normal_corr_07_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_07_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_m07":
    cfg = additive_1x2_normal_corr_m07_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_m07_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_095":
    cfg = additive_1x2_normal_corr_095_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_095_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x2_normal_corr_m095":
    cfg = additive_1x2_normal_corr_m095_config.cfg
    Net = additive_net.Net
    Generator = normal_corr_m095_generator.Generator
    Trainer = trainer.Trainer
    
else:
    print("None selected")
    sys.exit(0)
    

net = Net(cfg, "test")
generator = Generator(cfg, 'test')
m = Trainer(cfg, "test", net)
m.test(generator)
