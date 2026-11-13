"""One bounded raw-frame lookahead within the current causal QA cutoff."""
import bisect
import time
from concurrent.futures import ThreadPoolExecutor
from .stream_frames import decode_window


def decoded_windows(root, plan, times, start, end, window_frames, prefetch=False):
    ranges=[]
    while start < end:
        stop=min(end,start+window_frames,bisect.bisect_left(times,times[start]+window_frames/2))
        if (stop-start)%2 and stop<end:stop+=1
        stop=max(start+1,stop)
        ranges.append((start,stop));start=stop
    def decode(pair):
        a,b=pair;t=time.perf_counter();frames,metadata=decode_window(root,plan[a:b])
        return a,b,frames,metadata,time.perf_counter()-t
    if not prefetch:
        for pair in ranges:yield decode(pair)
        return
    if not ranges:return
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='causal-raw-decode') as pool:
        future=pool.submit(decode,ranges[0])
        for i in range(len(ranges)):
            result=future.result()
            future=pool.submit(decode,ranges[i+1]) if i+1<len(ranges) else None
            yield result
