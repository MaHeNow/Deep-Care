import torch
import os
from deepcare.models.conv_net import *

if __name__ == "__main__":

    model_path = "/home/mnowak/data/trained_models"
    model_name = "conv_net_w221_h221_v5/BalancedHmnChr14DSet/conv_net_v5_state_dict"
    model = conv_net_w221_h221_v5()

    state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(state_dict)

    model.eval()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save('/home/mnowak/data/trained_models/conv_net_w221_h221_v5/BalancedHmnChr14DSet/script_module_cpu.pt')
