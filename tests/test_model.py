import sys
sys.path.append("C:\\Users\\HC\\project\\LFM")
from models.LFM import SVDModel
import tensorflow as tf

import hydra
from omegaconf import DictConfig

def test_model():
    @hydra.main(config_path="../config", config_name="main.yaml", version_base="1.3.2")
    def _main(cfg: DictConfig):
        enc = hydra.utils.instantiate(cfg)
        model = SVDModel(enc.models)

        user_input = tf.constant([1.0, 2.0, 3.0,4.0])
        item_input =  tf.constant([4.0, 5.0, 6.0,7.0])  
        output_star = model([user_input, item_input])


