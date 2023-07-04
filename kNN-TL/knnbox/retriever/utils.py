r""" some utils function used for retrieve """
import torch
from knnbox.common_utils import global_vars
import numpy as np

def retrieve_k_nearest(query, faiss_index, k):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: i dont know why can't use view but must use reshape here 
    distances, indices = faiss_index.search(
                        query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k)
    # if "keys_count" not in global_vars():
    #     global_vars()["keys_count"] = 0
    # if "keys_vector" not in global_vars():
    #     global_vars()["keys_vector"] = query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy()
    # else:
    #     global_vars()["keys_vector"] = np.vstack((global_vars()["keys_vector"],query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy()))
    # global_vars()["keys_count"]+= len(query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy())
    
    distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices,device=query.device).view(*query_shape[:-1], k)

    return {"distances": distances, "indices": indices}
