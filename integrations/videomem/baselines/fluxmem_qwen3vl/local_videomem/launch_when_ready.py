"""Durable, one-shot small validation launcher. Never trains or stops other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'local_runs/smoke_v1'
DOWNLOAD=Path('/home/kai/models/.downloads/qwen25vl_7b_fluxmem')


def status(**row):
    row['updated_at_unix']=time.time()
    temp=WORK/'launcher_status.tmp'
    temp.write_text(json.dumps(row,indent=2)+'\n')
    temp.replace(WORK/'launcher_status.json')


def main(max_side=448):
    WORK.mkdir(parents=True,exist_ok=True)
    lock=WORK/'launcher.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('This one-shot launcher was already started; inspect saved results')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    deadline=time.monotonic()+7200
    while not (DOWNLOAD/'verification.json').exists():
        current=json.loads((DOWNLOAD/'status.json').read_text())
        status(phase='waiting_weights',download=current)
        if current['phase']=='failed':raise RuntimeError('Model download failed; see original download log')
        if time.monotonic()>deadline:raise TimeoutError('Weights did not finish within two hours')
        time.sleep(10)
    if not (WORK/'offline_ready.json').exists():
        raise RuntimeError('Offline setup/tests must pass before any GPU validation')
    jobs=[]
    for gpu,mode in [(4,'on'),(5,'off')]:
        destination=WORK/('fluxmem' if mode=='on' else 'uncompressed_control')
        log=open(WORK/f'gpu{gpu}.log','ab',buffering=0)
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'local_videomem/run.py'),
            '--manifest',str(WORK/'manifest.json'),'--questions',str(WORK/'questions.json'),
            '--output',str(destination),'--gpu',str(gpu),'--fluxmem',mode,
            '--fps','1','--max-frames','96','--max-side',str(max_side),'--max-new-tokens','256'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,gpu,mode));log.close()
    while any(p.poll() is None for p,_,_ in jobs):
        status(phase='running_validation',jobs=[{'pid':p.pid,'gpu':gpu,'fluxmem':mode,'exit_code':p.poll()}
                                             for p,gpu,mode in jobs])
        time.sleep(10)
    status(phase='finished' if all(p.returncode==0 for p,_,_ in jobs) else 'validation_error',
        jobs=[{'pid':p.pid,'gpu':gpu,'fluxmem':mode,'exit_code':p.returncode} for p,gpu,mode in jobs])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--max-side',type=int,default=448)
    args=parser.parse_args()
    WORK=args.work.resolve()
    try:main(args.max_side)
    except Exception as e:
        WORK.mkdir(parents=True,exist_ok=True)
        status(phase='launcher_error',error_type=type(e).__name__,error=str(e))
        raise
