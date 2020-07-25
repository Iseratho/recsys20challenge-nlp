from pathlib import PosixPath
import dask
import dask.dataframe as dd

def main():
    data_dir = PosixPath("~/recsys2020").expanduser()
    ds_name = "user_sampled"
    input_file = data_dir / f"{ds_name}_embeddings.parquet/"
    output_file = data_dir / f"{ds_name}_userembeddings.parquet/"
    df = dd.read_parquet(str(input_file))
    print(df.columns)
    df.groupby('user_id').agg({'embeddings': 'mean'})
    df.to_parquet(output_file)

if __name__ == "__main__":
    main()
