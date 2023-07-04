import faiss
import torch
import numpy as np
import os
import time

from knnbox.datastore import Datastore
from knnbox.retriever.utils import retrieve_k_nearest


class DatastoreCanLoadPartialIndex(Datastore):
    def load_faiss_index_subset(self, filename, subset_ids, move_to_gpu=True, verbose=True):
        r"""
        load faiss index from disk

        Args:
            filename: the prefix of faiss_index file, for example `keys.faiss_index`, filename is `keys`
            move_to_gpu: wether move the faiss index to GPU
        """
        index_path = os.path.join(self.path, filename+".faiss_index")
        
        if not hasattr(self, "faiss_index") or self.faiss_index is None:
            self.faiss_index = {}
        self.faiss_index[filename] = _load_faiss_index_subset(
                        path = index_path,
                        n_probe = 32,
                        subset_ids=subset_ids,
                        move_to_gpu = move_to_gpu,
                        verbose=verbose
                        )


def _load_faiss_index_subset(
    path, n_probe, subset_ids, move_to_gpu=True, verbose=True,
):
    print("[Start Loading Faiss Index]")
    if verbose:
        start_time = time.time()
    
    # check if the faiss index has been built
    if not os.path.exists(path):
        print("!!Error: faiss index hasn't beed built, Pleast built it first and then load it")
        import sys
        sys.exit(1)
 
    index = faiss.read_index(path, faiss.IO_FLAG_ONDISK_SAME_DIR)
    # not remove subset_ids
    sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(subset_ids))
    index.remove_ids(sel)
    if move_to_gpu:
        if verbose:
            print("  > move faiss index to gpu")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    if verbose:
        print("  > reading index took {} s".format(time.time()-start_time))
        print("  > the datastore shape is ", [index.ntotal, index.d])
    index.nprobe = n_probe
    print("[Finish Loading Faiss Index Successfully ^_^]")
    return index



if __name__ == "__main__":

    ds_full = DatastoreCanLoadPartialIndex.load(
        path="/home/user/yc27405/LowResource/ConsisitTLknn/datastore/visual-child",
        load_list=["vals"],)
    ds_full.load_faiss_index("keys")
    
    # construct a pseduo batched query
    query = torch.randn(100, 512)
    
    start = time.time()
    for _ in range(1000):
        result = retrieve_k_nearest(query, ds_full.faiss_index["keys"], k=8)
    print("retrieve result of entire index: ", result["indices"][0])
    print("Time elapsed: ", time.time() - start)
    print("\n---------------------------------")

    ds = DatastoreCanLoadPartialIndex.load(
        path="/home/user/yc27405/LowResource/ConsisitTLknn/datastore/visual-child",
        load_list=["vals"],)
    # construct a pseduo subset
    subset_ids = np.arange(5000, 10000) # subset_ids should be 1D numpy array
    ds.load_faiss_index_subset("keys", subset_ids)
    print("only load index size: ", ds.faiss_index["keys"].ntotal)


    start = time.time()
    for _ in range(1000):
        result = retrieve_k_nearest(query, ds.faiss_index["keys"], k=8)
    print("retrieve result of partial index: ", result["indices"][0])
    print("Time elapsed: ", time.time() - start)
