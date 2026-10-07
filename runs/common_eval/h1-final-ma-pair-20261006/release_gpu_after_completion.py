#!/usr/bin/env python3
"""Release only this MA run's 8001 model server after all requested arms complete."""
from __future__ import annotations
import json
import os
import signal
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STATUS=ROOT/'orchestrator-status.json'
RUNTIME=ROOT/'vllm-runtime-config.json'
RELEASE=ROOT/'gpu-release.json'
LOG=ROOT/'gpu-release.log'


def note(message: str) -> None:
    line=f"{datetime.now(UTC).isoformat()} {message}\n"
    with LOG.open('a',encoding='utf-8') as out:
        out.write(line); out.flush()
    print(line,end='',flush=True)


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(('127.0.0.1',port))==0


def main() -> int:
    note('watching final B0/B2 orchestrator; no process signal will be sent before COMPLETE')
    while True:
        try:
            state=json.loads(STATUS.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):
            time.sleep(30); continue
        status=state.get('status')
        if status=='FAILED':
            note('orchestrator FAILED; retaining 8001 for checkpoint recovery')
            return 1
        if status=='COMPLETE':
            break
        time.sleep(30)
    runtime=json.loads(RUNTIME.read_text(encoding='utf-8'))
    pid=int(runtime['current_server_pid'])
    cmdline=Path(f'/proc/{pid}/cmdline')
    try:
        command=cmdline.read_bytes()
    except OSError:
        command=b''
    if b'--port\x008001\x00' not in command or b'vllm' not in command:
        note(f'refusing to signal PID {pid}: it is not the recorded 8001 vLLM server')
        return 2
    os.kill(pid,signal.SIGTERM)
    note(f'SIGTERM sent only to recorded 8001 server PID {pid}; port 8000 is untouched')
    deadline=time.monotonic()+60
    while time.monotonic()<deadline and port_open(8001):
        time.sleep(2)
    result={
      'schema_version':'h1-ma-gpu-release-v1',
      'released_at_utc':datetime.now(UTC).isoformat(),
      'orchestrator_status':status,
      'stopped_pid':pid,
      'port_8001_closed':not port_open(8001),
      'port_8000_left_untouched':True,
      'handoff':'GPU available for medical SFT/GSPO/GDPO after final B0/B2 closeout',
    }
    RELEASE.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    note('release record written: '+json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0 if result['port_8001_closed'] else 3

if __name__=='__main__':
    raise SystemExit(main())
