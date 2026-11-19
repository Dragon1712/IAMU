from __future__ import annotations
import queue
import threading
import unittest
from dataclasses import dataclass
from v11_protocol.dispatch_v12 import RoundDispatcher

@dataclass(frozen=True)
class Job:
    episode:int

class TestDispatchV12(unittest.TestCase):
    def test_out_of_order_results_all_assigned_exactly_once(self):
        q=queue.Queue()
        commands=[queue.Queue() for _ in range(3)]
        jobs=[Job(i) for i in range(1,17)]
        dispatcher=RoundDispatcher(q,commands,jobs)
        dispatcher.start()
        received=[]
        # Simulate one result for each job in shuffled processing order; the
        # coordinator must keep dispatching and never corrupt episode identity.
        in_process=[]
        for worker, cmd in enumerate(commands):
            in_process.append((worker,cmd.get_nowait()[1]))
        while len(received)<len(jobs):
            worker,job=in_process.pop(-1)  # explicitly non-FIFO completion
            q.put(('ok',{'episode':job.episode,'worker_id':worker},[]))
            received.append(dispatcher.get(timeout=3)[1]['episode'])
            try: in_process.append((worker, commands[worker].get(timeout=.5)[1]))
            except queue.Empty: pass
        dispatcher.finish()
        self.assertEqual(set(received),set(range(1,17)))
        self.assertEqual(len(received),len(set(received)))

if __name__=='__main__':unittest.main()
