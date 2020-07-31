import gzip
import argparse
import dask
import dask.dataframe as dd

def handle_file(filepath: str):
    df = dd.read_table(filepath, sep='\x01')

    print(f'Usual length: {usual_length}')
    print(f'Num lines: {num_lines}')
    print(f'Num tweet ids: {len(tweet_id_set)}')
    print(f'Num engaging users: {len(engaging_user_id_set)}')
    print(f'Num engaged users: {len(engaged_with_user_id_set)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()

    filepath = args.filepath
    handle_file(filepath)


if __name__ == "__main__":
    main()
