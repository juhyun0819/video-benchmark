"""Durable pipeline: verified runtime -> audio -> real smoke -> full caption job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from .archive import atomic_json


def choose_smoke_clips(clips):
    if not clips:
        raise ValueError('Empty clip inventory')
    first_day = clips[0]['day_id']
    first_day_clips = [c for c in clips if c['day_id'] == first_day]
    selected = first_day_clips[:2] + first_day_clips[-1:] + clips[-1:]
    return list(dict.fromkeys(c['clip_id'] for c in selected))


def launch(inventory, runtime_path, output, audio_cache, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Provide distinct nonnegative physical GPU indices')
    inventory=Path(inventory).resolve()
    runtime_path=Path(runtime_path).resolve()
    audio_cache=Path(audio_cache).resolve()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    smoke = root.with_name(root.name + '_smoke')
    logs = root/'logs';logs.mkdir(exist_ok=True)
    children = []
    phase = 'starting'
    repository = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONNOUSERSITE':'1', 'OMP_NUM_THREADS':'4',
           'TOKENIZERS_PARALLELISM':'false'}

    def status(**data):
        atomic_json(root/'pipeline.json',{'phase':phase,'pid':os.getpid(),
                    'updated_unix':time.time(),'runtime':str(runtime_path),
                    'output':str(root),'smoke_output':str(smoke),'gpus':gpus,**data})

    def start(args, label):
        with (logs/(label+'.log')).open('a') as log:
            process = subprocess.Popen(args, cwd=repository, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        children.append(process)
        return process

    def wait(process, label):
        while process.poll() is None:
            status(active_step=label,child_pid=process.pid)
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f'{label} failed with exit {process.returncode}; see logs/{label}.log')

    def run(args, label):
        wait(start(args,label),label)

    def await_runtime(media_only=False):
        while True:
            try:
                runtime = json.loads(Path(runtime_path).read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                runtime = {}
            if runtime.get('status') in ('error','failed'):
                raise RuntimeError('Runtime preparation failed; inspect runtime logs')
            if runtime.get('status') == 'ready' or (media_only and runtime.get('media_ready')):
                return runtime
            status(runtime_status=runtime.get('status','unavailable'))
            time.sleep(15)

    def interrupted(signum, frame):
        raise InterruptedError(f'Pipeline interrupted by signal {signum}')

    with (root/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        try:
            phase='waiting_for_audio_runtime'
            runtime=await_runtime(media_only=True)
            python=runtime['python']
            command=[python,'-u','-m','baselines.egogpt_caption']
            phase='preparing_native_audio'
            run(command+['audio-cache','--inventory',str(inventory),'--output',str(audio_cache)],'audio-cache')
            phase='waiting_for_verified_model'
            runtime=await_runtime()
            if python != runtime['python']:
                raise ValueError('Runtime interpreter changed during audio preparation')
            phase='preparing_smoke'
            run(command+['prepare','--inventory',str(inventory),'--runtime',str(runtime_path),
                         '--output',str(smoke),'--workers','1'],'smoke-prepare')
            smoke_manifest=json.loads((smoke/'manifest.json').read_text())
            selected=choose_smoke_clips(smoke_manifest['clips'])
            phase='smoke_inference'
            run(command+['worker','--output',str(smoke),'--index','0','--gpu',str(gpus[0]),
                         '--audio-cache',str(audio_cache),'--port','29680','--clip-ids',*selected],
                'smoke-worker')
            phase='verifying_smoke'
            run([python,'-m','baselines.egogpt_caption.audit','--output',str(smoke),
                 '--smoke','--clip-ids',*selected],'smoke-audit')
            phase='preparing_full_job'
            run(command+['prepare','--inventory',str(inventory),'--runtime',str(runtime_path),
                         '--output',str(root),'--workers',str(len(gpus))],'full-prepare')
            phase='captioning'
            workers=[]
            for index,gpu in enumerate(gpus):
                workers.append(start(command+['worker','--output',str(root),'--index',str(index),
                    '--gpu',str(gpu),'--audio-cache',str(audio_cache),'--port',str(29690+index)],
                    f'worker_{index:02d}'))
                # Stagger native checkpoint loads to reduce simultaneous host allocations.
                time.sleep(5)
            while any(p.poll() is None for p in workers):
                status(worker_pids=[p.pid for p in workers],
                       worker_exit_codes=[p.poll() for p in workers],
                       receipt_files=len(list((root/'receipts').glob('*.json'))))
                time.sleep(30)
            phase='auditing_full_job'
            run([python,'-m','baselines.egogpt_caption.audit','--output',str(root)],'full-audit')
            if any(p.returncode for p in workers):
                raise RuntimeError('A worker exited unsuccessfully; inspect worker logs')
            phase='complete';status()
        except BaseException as exc:
            failed_phase=phase
            phase='error';status(failed_phase=failed_phase,error=str(exc),traceback=traceback.format_exc())
            raise
        finally:
            for process in children:
                if process.poll() is None:
                    process.terminate()
            for process in children:
                if process.poll() is None:
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','runtime','output','audio-cache']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    args=parser.parse_args()
    launch(args.inventory,args.runtime,args.output,args.audio_cache,args.gpus)


if __name__=='__main__':main()
