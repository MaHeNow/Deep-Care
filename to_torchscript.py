import torch
import os
from deepcare.models.conv_net import *

if __name__ == "__main__":

    model_path = "/home/mnowak/data/trained_models/conv_net_w224_h224_v5/LargerBalancedHmnChr14DSet/conv_net_v4_state_dict"
    model = conv_net_w224_h224_v5()

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save('/home/mnowak/data/trained_models/conv_net_w224_h224_v5/LargerBalancedHmnChr14DSet/script_module.pt')
