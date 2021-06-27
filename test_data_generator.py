from util.data_frame_image_data_generator import DataFrameImageDataGenerator, load_image
from util.dataset_data_frame_generator import generate_data_frame
from util.profiling import start_profiling

def setup_generator(df, batch_size, do_parallel):
    return DataFrameImageDataGenerator(df, batch_size,
        n_classes=df.movielens_id.nunique(),
        use_ratings=True,
        use_genres=True,
        use_class=True,
        use_self_supervised=False,
        do_parallel=do_parallel
    )

if __name__ == '__main__':
    batch_size = 128
    do_parallel=True

    df = generate_data_frame()

    # print('DO use Async')
    # gen = setup_generator(df, batch_size, True)
    # start_profiling(gen.__getitem__, 0)

    # print('DO NOT use Async')
    # gen = setup_generator(df, batch_size, False)
    # start_profiling(gen.__getitem__, 0)

    gen = setup_generator(df, batch_size, do_parallel)
    start_profiling(gen.__getitem__, 0, print_stats=True, save_stats=False)
    # gen.__getitem__(0)
    # start_profiling(load_image, df.iloc[0]['full_path'])
