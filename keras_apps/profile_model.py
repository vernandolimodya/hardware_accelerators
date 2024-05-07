import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

from datetime import datetime

def op_stats(model):
    op_stats = {'layer_name': [],
                'layer_type': [],
                'MACs': [],
                'bias_adds': [],
                'max/avg': [],
                'flops': [],
                'activations': []
                }

    for layer in model.layers:

        op_stats['layer_name'].append(layer.name)
        op_stats['layer_type'].append(layer.__class__.__name__)

        if layer.__class__.__name__ == 'Conv2D':
            macs = int(layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[3] * layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3])
            op_stats['MACs'].append(macs)
            bias_adds = int(layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3])
            op_stats['bias_adds'].append(bias_adds)

            op_stats['max/avg'].append(0)

            op_stats['flops'].append(2*macs + bias_adds)

        elif layer.__class__.__name__ == 'MaxPooling2D':
            op_stats['MACs'].append(0)
            op_stats['bias_adds'].append(0)
            max_avg = int(layer.pool_size[0]*layer.pool_size[1]*layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3])
            op_stats['max/avg'].append(max_avg)

            op_stats['flops'].append(layer.pool_size[0]*layer.pool_size[1]*layer.output_shape[1]*layer.output_shape[2]*layer.output_shape[3])
        
        elif layer.__class__.__name__ == 'Dense':
            macs = int(layer.input_shape[1]*layer.output_shape[1])
            op_stats['MACs'].append(macs)
            bias_adds = int(layer.output_shape[1])
            op_stats['bias_adds'].append(bias_adds)

            op_stats['max/avg'].append(0)

            op_stats['flops'].append(2*macs + bias_adds)
        
        else:
            op_stats['MACs'].append(np.nan)
            op_stats['bias_adds'].append(np.nan)
            op_stats['max/avg'].append(np.nan)
            op_stats['flops'].append(np.nan)

        if 'activation' in layer.get_config().keys():
            op_stats['activations'].append(layer.get_config()['activation'])
        else:
            op_stats['activations'].append(np.nan)
        

    return pd.DataFrame.from_dict(op_stats)