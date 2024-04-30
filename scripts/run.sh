#!/usr/bin/bash
accelerate launch --multi_gpu  --config_file scripts/accelerate_config/single_node_multi_gpu.yaml pretrain/pretrain.py