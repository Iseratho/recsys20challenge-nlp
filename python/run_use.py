import numpy as np
import pyarrow
import pyarrow.parquet as pq
from pathlib import PosixPath
from functools import partial
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, BertTokenizer
import torch
import tqdm
import shutil
import pandas as pd
import math


def main():
    # print(pyarrow.array(np.arange(0,10, 0.1).tolist()).type)
    data_dir = PosixPath("~/recsys2020").expanduser()
    input_file = data_dir / "training1m.parquet"
    output_file = data_dir / "training1m_stage1.parquet"

    if output_file.is_dir:
        shutil.rmtree(output_file)

    output_file.mkdir()

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    d =  torch.device("cuda")
    max_token_len = 600

    with torch.no_grad():
        model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased').to(d)
        calc_embeddings = model.get_input_embeddings()


        for partition_file in input_file.glob("*.parquet"):
            print(f"Processing {partition_file.name}")
            table = pd.read_parquet(partition_file)

            res = []
            num_rows = table.shape[0]
            batch_size = 15
            num_batches = int(math.ceil(num_rows / batch_size))
            with tqdm.tqdm(total=num_rows) as p:
                for b_idx in range(num_batches):
                    int_col = table['tokens'].iloc[b_idx:b_idx+batch_size]
                    pad_len = min(max_token_len, max(len(a) for a in int_col))
                    int_col = [a[:pad_len].tolist() if len(a) >= pad_len else a.tolist() + (pad_len - len(a)) * [0] for a in int_col]
                    input_ids = torch.tensor(int_col).to(d)
                    mask = (input_ids != 0).unsqueeze(2)
                    outputs =   (
                                    (calc_embeddings(input_ids) * mask)
                                    .sum(axis=1) / mask.sum(axis=1)
                                ).cpu().numpy()
                    for i in range(batch_size):
                        res.append(outputs[i].astype(np.float32))
                    # result_embeddings.append(outputs)
                    p.update(len(int_col))
            table["embeddings"] = pd.Series(res)
            print(table.columns)
            table.columns = table.columns.astype(str)
            table.to_parquet(output_file / partition_file.name)







if __name__ == "__main__":
    main()