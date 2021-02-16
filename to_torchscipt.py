import torch
import os
from deepcare.models.conv_net import \
    conv_net_w51_h100_v1, \
    conv_net_w51_h100_v2, \
    conv_net_w51_h100_v3, \
    conv_net_w51_h100_v4, \
    conv_net_w51_h100_v5, \
    conv_net_w51_h100_v6, \
    conv_net_w51_h100_v7, \
    conv_net_w51_h100_v8, \
    conv_net_w51_h100_v9, \
    conv_net_w51_h100_v10, \
    conv_net_w250_h50_v1

if __name__ == "__main__":

    model_path = "trained_models"
    model_name = "conv_net_v10_w51_h100/AthalianaElegansMix/conv_net_v10_state_dict"
    model = conv_net_w51_h100_v10()

    state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(state_dict)

    model.eval()

    traced_script_module = torch.jit.script(model)
    traced_script_module.save('athaliana_elegans_mix_model.pt')
