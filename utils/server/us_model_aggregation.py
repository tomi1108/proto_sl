import torch.nn as nn
from typing import Dict, List

def model_aggregation(
    client_model: nn.Sequential,
    client_model_dict: Dict[str, nn.Sequential],
    weights_per_client: List[float] 
) -> nn.Sequential:

    for clt in range(len(client_model_dict)):

        add_client_model = client_model_dict['Client {}'.format(clt+1)]
        weight = weights_per_client[clt]

        for param1, param2 in zip(client_model.parameters(), add_client_model.parameters()):

            add_param = param2.data * weight
    
            if clt == 0:

                param1.data.copy_(add_param)
            
            else:

                param1.data.add_(add_param)
    
    return client_model


# fed_flag == False の場合の関数を実装する