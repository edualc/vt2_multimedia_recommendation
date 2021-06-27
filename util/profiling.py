import os
import cProfile
import pstats
from datetime import datetime

# import asyncio

# lehl@2021-06-26: Profiling function built, inspired by the 
# mCoding video about profiling in Python:
# --> https://www.youtube.com/watch?v=m_a0fN48Alw
# 
# Run "snakeviz <profiling_file>" to view a visualization
#
def start_profiling(func, *args, print_stats=False, save_stats=True, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.sort_stats('time')
    # stats.sort_stats(pstats.SortKey.TIME) #py3.7+

    if print_stats:
        stats.print_stats(16)

    if save_stats:
        time_stamp = datetime.now().strftime('%Y_%m_%d__%H%M%S')
        file_name = '_profiling_dumps/' + '_'.join([time_stamp, func.__name__ + '.prof'])
        stats.dump_stats(file_name)

if __name__ == '__main__':
    def do_stuff():
        x = np.arange(2**20)
        y = np.arange(2**22)
        z = np.intersect1d(x,y)

    import numpy as np
    start_profiling(do_stuff)
