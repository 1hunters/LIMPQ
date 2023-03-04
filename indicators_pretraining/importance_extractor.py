import torch
import pickle

checkpoint_path = '/data/ctang/saved_results/LIMPQ/checkpoints/resnet50_65432_checkpoint.pth.tar'
importance_indicators_save_path = 'importance_indicators.pkl'

cand = [6, 5, 4, 3, 2] # same bit-width candidates in the indicator pretraining YAML file. 

state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']

act, weight, layer_name = {}, {}, []

for k, v in state_dict.items():
    k: str

    if k.endswith('.quan_a_fn.s'):
        layer = k.split('.quan_a_fn.s')[0]
        assert layer not in act
        act[layer] = {}

        for bit_witdh, activation_scale_factor_value in zip(cand, v):
            # print(activation_scale_factor_value)
            act[layer][str(bit_witdh)] = activation_scale_factor_value.item()
    
    if k.endswith('.quan_w_fn.s'):
        layer = k.split('.quan_w_fn.s')[0]
        assert layer not in weight
        weight[layer] = {}

        for bit_witdh, weight_scale_factor_value in zip(cand, v):
            weight[layer][str(bit_witdh)] = weight_scale_factor_value.item()
        
        layer_name.append(layer)
    

with open(importance_indicators_save_path, 'wb') as f:
    pickle.dump({
        'weight': weight,
        'act': act,
        'layer_name': layer_name
    }, f)