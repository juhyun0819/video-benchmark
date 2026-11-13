from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess

import numpy as np
import pytest
from baselines.qwen3_offline.video import build_inventory
from baselines.egogpt_caption import media


def make_video(path, *, frames=76, fps=2, audio=True):
    video_path=path.with_name(path.stem+'.video.mp4') if audio else path
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=64x48:rate={fps}',
                    '-frames:v',str(frames),'-c:v','libx264','-g','1','-bf','0',str(video_path)],check=True)
    if audio:
        subprocess.run(['ffmpeg','-v','error','-i',str(video_path),'-f','lavfi','-i',
                        'sine=frequency=440:sample_rate=48000','-c:v','copy','-c:a','aac',
                        '-t',str(float(Fraction(frames,fps))),str(path)],check=True)
    return path


@pytest.fixture
def inventory(tmp_path):
    paths=[make_video(tmp_path/'day1.mp4', frames=63), make_video(tmp_path/'day2.mp4', frames=60)]
    return build_inventory([str(p) for p in paths])


def test_clips_include_last_partial_and_never_cross_day(inventory):
    clips=media.build_clips(inventory)
    assert len(clips)==3
    assert [c['clip_id'] for c in clips]==['day1_000000','day1_000001','day2_000000']
    assert [(c['source_start_fraction'],c['source_end_fraction']) for c in clips]==[('0','30'),('30','63/2'),('0','30')]
    assert [(c['global_start_fraction'],c['global_end_fraction']) for c in clips]==[('0','30'),('30','63/2'),('63/2','123/2')]
    assert clips[0]['frame_indices']==list(range(0,60,2))
    assert clips[1]['frame_indices']==[60,62]
    assert clips[2]['frame_indices']==list(range(63,123,2))
    assert clips[1]['duration_fraction']=='3/2'


def test_decode_original_rgb_and_16khz_audio_with_partial_clip(inventory):
    clip=media.build_clips(inventory)[1]
    frames,audio,metadata=media.decode_clip(inventory,clip)
    assert frames.shape==(2,48,64,3)
    assert frames.dtype==np.uint8
    assert audio.ndim==1 and audio.dtype==np.float32
    assert len(audio)==24000
    assert np.isfinite(audio).all() and np.abs(audio).max()>0.01
    assert metadata['audio']['sample_rate']==16000
    assert metadata['audio']['sample_count']==24000
    assert [x['source_seconds_fraction'] for x in metadata['frames']]==['30','31']
    assert all(Fraction(x['global_seconds_fraction'])<Fraction(clip['global_end_fraction']) for x in metadata['frames'])


def test_full_clip_audio_is_not_whisper_padded_or_trimmed(tmp_path):
    inv=build_inventory([str(make_video(tmp_path/'tiny.mp4',frames=3,fps=2))])
    frames,audio,metadata=media.decode_clip(inv,media.build_clips(inv)[0])
    assert frames.shape[0]==2
    assert len(audio)==24000
    assert metadata['audio']['duration_fraction']=='3/2'


def test_missing_audio_fails_instead_of_silent_zero_audio(tmp_path):
    inv=build_inventory([str(make_video(tmp_path/'silent.mp4',frames=4,audio=False))])
    with pytest.raises(ValueError,match='audio'):
        media.decode_clip(inv,media.build_clips(inv)[0])


@pytest.mark.parametrize('field,value',[
    ('frame_indices',[0,62]),('source_end_fraction','32'),('global_end_fraction','32'),
    ('clip_id','wrong'),('path','/tmp/other.mp4'),('day_id','Day2'),
])
def test_modified_clip_cannot_change_observation_range(inventory,field,value):
    clip=deepcopy(media.build_clips(inventory)[0]);clip[field]=value
    with pytest.raises(ValueError):media.decode_clip(inventory,clip)


def test_modified_source_rejected_before_read(inventory):
    clip=media.build_clips(inventory)[0]
    with open(clip['path'],'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        media.decode_clip(inventory,clip)


def test_noninteger_fps_rejected_for_exact_1fps_sampling(tmp_path):
    inv=build_inventory([str(make_video(tmp_path/'fractional.mp4',frames=60,fps=Fraction(3,2)))])
    with pytest.raises(ValueError,match='integer'):
        media.build_clips(inv)


def test_audio_cache_matches_original_full_load_then_slice(inventory,tmp_path):
    import librosa
    clip=media.build_clips(inventory)[1]
    directory=tmp_path/'audio-cache'
    manifest=media.prepare_audio_cache(inventory,directory)
    assert len(manifest)==2
    _,direct,_=media.decode_clip(inventory,clip)
    _,cached,metadata=media.decode_clip(inventory,clip,audio_cache_dir=directory)
    native,_=librosa.load(clip['path'],sr=16000)
    assert np.array_equal(direct,native[480000:504000])
    assert np.array_equal(cached,direct)
    assert metadata['audio']['cached'] is True
    # Reusing an unchanged cache must preserve the exact artifact.
    before=(directory/'day1.npy').stat().st_mtime_ns
    media.prepare_audio_cache(inventory,directory)
    assert (directory/'day1.npy').stat().st_mtime_ns==before


def test_modified_audio_cache_rejected(inventory,tmp_path):
    directory=tmp_path/'audio-cache'
    media.prepare_audio_cache(inventory,directory)
    with open(directory/'day1.npy','ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='cache'):
        media.decode_clip(inventory,media.build_clips(inventory)[0],audio_cache_dir=directory)
