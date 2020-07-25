import numpy as np
import pyarrow
import pyarrow.parquet as pq
from pathlib import PosixPath
from functools import partial
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, BertTokenizer, BertModel
import torch
import tqdm
import shutil
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
import math
import torch.multiprocessing as mp

max_token_len = 600
arch = 'distilbert'
# arch = 'bert'

def embed_partition(df: dd.DataFrame,d, model) -> dd.DataFrame:
    num_rows = df.shape[0]
    batch_size = 4
    num_batches = int(math.ceil(num_rows / batch_size))
    res = []
    with tqdm.tqdm(total=num_rows, position=0) as p:
        for b_idx in range(num_batches):
            int_col = df.loc[b_idx:b_idx+batch_size, 'tokens']
            # # GPU-side padding is even slower:
            # tensors = [torch.tensor(i).to(d) for i in int_col]
            # input_ids = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
            # input_ids = input_ids.long()

            # CPU/Numpy seems to be better at padding:
            pad_len = min(max_token_len, max(len(a) for a in int_col))
            int_col = np.vstack([np.pad(i, (0, pad_len - len(i),), constant_values=(1,)) if len(i) <
                    pad_len else i[:pad_len] for i in int_col])
            input_ids = torch.tensor(int_col).to(d).long()
            mask = (input_ids != 0).unsqueeze(2)
            outputs =   (
                    (torch.stack(model(input_ids)[2 if arch == 'bert' else 1][:-2], dim=0).sum(axis=0) * mask)
                    .sum(axis=1) / mask.sum(axis=1)
                    ).float().cpu()
            res.extend(outputs[i].numpy() for i in range(outputs.shape[0]))
            p.update(len(int_col))
    df.loc[:, "embeddings"] = pd.Series(res)
    return df

def main():
    with dask.config.set(scheduler='synchronous'):
        data_dir = PosixPath("~/recsys2020").expanduser()
        ds_name = "user_sampled"
        input_file = data_dir / f"{ds_name}.parquet/"
        output_file = data_dir / f"{ds_name}_embeddings.parquet/"
        df = dd.read_parquet(str(input_file))

        meta = {
            'user_id': str,
            'tweet_id': str,
            'tokens': object,
            'embeddings': object
        }
        d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            if arch == 'distilbert':
                model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased', output_hidden_states=True)
            elif arch == 'bert':
                model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
            model = model.eval().to(d)
            df = df[['user_id', 'tweet_id', 'tokens']].map_partitions(embed_partition, d=d, model=model, meta=meta)
            del df['tokens']
            df.to_parquet(output_file)

if __name__ == "__main__":
    main()
