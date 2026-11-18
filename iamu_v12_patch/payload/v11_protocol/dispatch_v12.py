"""Optional bounded, round-local actor dispatcher; never touches learner or Replay.

The coordinator assigns the next episode when a worker finishes, even while
main thread performs an ordered gradient update. It only sees immutable jobs and
actor result messages. Episode ID consumption remains exclusively in the learner.
"""
from __future__ import annotations
import queue
import threading
from collections import deque

class RoundDispatcher:
    def __init__(self, result_queue, commands, batch):
        self.result_queue=result_queue
        self.commands=commands
        self.waiting=deque(batch)
        self.expected={j.episode for j in batch}
        if len(self.expected)!=len(batch): raise ValueError('Duplicate episode IDs in round')
        self.seen=set()
        self.completed=queue.Queue(maxsize=max(1,len(batch)))
        self.cancelled=threading.Event()
        self.thread=threading.Thread(target=self._run,name='v12-round-dispatcher',daemon=True)
        self.error=None

    def _assign(self, worker_id):
        if self.waiting:
            job=self.waiting.popleft()
            self.commands[worker_id].put(('rollout',job),timeout=30)

    def start(self):
        for worker in range(len(self.commands)):
            self._assign(worker)
        self.thread.start()

    def _run(self):
        try:
            while not self.cancelled.is_set() and len(self.seen)<len(self.expected):
                try: msg=self.result_queue.get(timeout=1)
                except queue.Empty: continue
                tag=msg[0]
                if tag=='error':
                    self.completed.put(msg)
                    return
                if tag!='ok': continue
                report=msg[1]
                episode=int(report['episode'])
                worker_id=int(report['worker_id'])
                if worker_id<0 or worker_id>=len(self.commands):
                    raise ValueError(f'Invalid worker ID: {worker_id}')
                if episode not in self.expected or episode in self.seen:
                    raise ValueError(f'Unexpected/duplicate actor completion: {episode}')
                self.seen.add(episode)
                self._assign(worker_id)
                self.completed.put(msg)
        except BaseException as exc:
            self.error=exc
            self.completed.put(('error',-1,repr(exc)))

    def get(self,timeout=30):
        return self.completed.get(timeout=timeout)

    def finish(self):
        self.thread.join(timeout=10)
        if self.thread.is_alive(): raise RuntimeError('Dispatcher did not stop after round')
        if self.error is not None: raise RuntimeError('Dispatcher failed') from self.error
        if self.waiting or len(self.seen)!=len(self.expected):
            raise RuntimeError('Dispatcher completed a partial round')

    def cancel(self):
        self.cancelled.set()
