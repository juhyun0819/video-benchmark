"""Optional local-checkpoint contract test; no weights or GPU required."""
import os
from fractions import Fraction
import subprocess
import numpy as np
import pytest
from baselines.qwen3_offline.video import build_inventory, decode_indices
from baselines.qwen3_offline.run import sample_prefix, process_frames, messages


@pytest.mark.skipif(not os.environ.get('QWEN3_MODEL'), reason='Set QWEN3_MODEL to a local official processor')
@pytest.mark.parametrize('split_days',[False,True])
def test_seek_prefix_inputs_equal_official_raw_file_path(tmp_path,split_days):
    import torch
    from transformers import AutoProcessor
    from transformers.video_utils import read_video_pyav
    original = tmp_path/'original.mp4'
    prefix = tmp_path/'prefix.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=25',
                    '-frames:v','100','-c:v','libx264','-g','1','-bf','0',str(original)],check=True)
    # Lossless packet copy: identical first 31 frames, bounded raw-video reference.
    subprocess.run(['ffmpeg','-v','error','-i',str(original),'-frames:v','31','-c:v','copy',str(prefix)],check=True)
    processor=AutoProcessor.from_pretrained(os.environ['QWEN3_MODEL'],local_files_only=True)
    paths=[str(original)]
    if split_days:
        day1,day2=tmp_path/'day1.mp4',tmp_path/'day2.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(original),'-frames:v','25','-c:v','copy',str(day1)],check=True)
        subprocess.run(['ffmpeg','-v','error','-ss','1','-i',str(original),'-c:v','copy',str(day2)],check=True)
        paths=[str(day1),str(day2)]
    inv=build_inventory(paths)
    count,indices,records=sample_prefix(processor,inv,Fraction('1.2'))
    frames,_=decode_indices(inv,indices,Fraction('1.2'))
    raw,metadata=read_video_pyav(str(prefix),processor.video_processor.sample_frames)
    assert count==31
    if split_days:
        assert records[-1]['day_id']=='Day2'
    assert metadata.frames_indices.tolist()==indices
    assert np.array_equal(np.stack(frames),raw)
    assert all(Fraction(r['global_seconds_fraction']) <= Fraction('1.2') for r in records)
    q={'question':'Name one visible color.','choices':[]}
    observed=process_frames(processor,frames,indices,count,inv['fps'],q)
    prompt=processor.apply_chat_template(messages(q),tokenize=False,add_generation_prompt=True)
    reference=processor(text=[prompt],videos=[raw],video_metadata=[metadata],do_sample_frames=False,return_tensors='pt')
    assert set(observed)==set(reference)
    for name in observed:
        assert torch.equal(observed[name],reference[name]), name
