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
import math
import torch.multiprocessing as mp


def main():
    # print(pyarrow.array(np.arange(0,10, 0.1).tolist()).type)
    data_dir = PosixPath("~/recsys2020").expanduser()
    ds_name = "training1m"
    input_file = data_dir / f"{ds_name}.parquet"
    output_file = data_dir / f"{ds_name}_stage1.parquet"
    partition_files = list(input_file.glob("*.parquet"))

    if output_file.exists():
        shutil.rmtree(output_file)

    output_file.mkdir()

    arch = 'distilbert'
    # arch = 'bert'

    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_token_len = 600

    with tqdm.tqdm(total=len(partition_files), position=1) as pp:
        with torch.no_grad():
            if arch == 'distilbert':
                model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased', output_hidden_states=True)
            elif arch == 'bert':
                model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
            model = model.eval().to(d)

            for partition_file in partition_files:
                # print(f"Processing {partition_file.name}")
                table = pd.read_parquet(partition_file)

                res = []
                num_rows = table.shape[0]
                batch_size = 50
                num_batches = int(math.ceil(num_rows / batch_size))
                with tqdm.tqdm(total=num_rows, position=0) as p:
                    for b_idx in range(num_batches):
                        int_col = table['tokens'].iloc[b_idx:b_idx+batch_size]
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
                table["embeddings"] = pd.Series(res)
                table.columns = table.columns.astype(str)
                table.to_parquet(output_file / partition_file.name)
                pp.update(1)







if __name__ == "__main__":
    main()
