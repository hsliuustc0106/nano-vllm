from types import SimpleNamespace
import unittest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def scheduler_config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
    )


class SchedulerCancelTests(unittest.TestCase):

    def test_cancel_waiting_sequence(self):
        scheduler = Scheduler(scheduler_config())
        seq = Sequence([1, 2, 3])
        scheduler.add(seq)

        cancelled = scheduler.cancel(seq.seq_id)

        self.assertIs(cancelled, seq)
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(seq.finish_reason, "cancelled")
        self.assertFalse(scheduler.waiting)

    def test_cancel_running_sequence_deallocates_blocks(self):
        scheduler = Scheduler(scheduler_config())
        seq = Sequence([1, 2, 3])
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        cancelled = scheduler.cancel(seq.seq_id)

        self.assertIs(cancelled, seq)
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(seq.finish_reason, "cancelled")
        self.assertEqual(seq.block_table, [])
        self.assertFalse(scheduler.running)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 4)

    def test_postprocess_sets_length_finish_reason(self):
        scheduler = Scheduler(scheduler_config())
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        appended = scheduler.postprocess([seq], [4], is_prefill=False)

        self.assertEqual(appended, [(seq, 4)])
        self.assertEqual(seq.finish_reason, "length")
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(seq.block_table, [])

    def test_postprocess_sets_stop_finish_reason_for_eos(self):
        scheduler = Scheduler(scheduler_config())
        seq = Sequence([1, 2, 3])
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        appended = scheduler.postprocess([seq], [-1], is_prefill=False)

        self.assertEqual(appended, [(seq, -1)])
        self.assertEqual(seq.finish_reason, "stop")
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertEqual(seq.block_table, [])

    def test_schedule_fills_available_sequence_slots_before_decode(self):
        scheduler = Scheduler(scheduler_config())
        running = Sequence([1, 2, 3])
        scheduler.block_manager.allocate(running, 0)
        running.status = SequenceStatus.RUNNING
        running.is_prefill = False
        scheduler.running.append(running)
        scheduler.last_schedule_was_prefill = True
        waiting = Sequence([4, 5, 6])
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertFalse(scheduler.waiting)
        self.assertEqual(list(scheduler.running), [running, waiting])

    def test_schedule_decodes_when_running_slots_are_full(self):
        scheduler = Scheduler(scheduler_config())
        scheduler.max_num_seqs = 1
        running = Sequence([1, 2, 3])
        scheduler.block_manager.allocate(running, 0)
        running.status = SequenceStatus.RUNNING
        running.is_prefill = False
        scheduler.running.append(running)
        scheduler.last_schedule_was_prefill = True
        waiting = Sequence([4, 5, 6])
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertEqual(list(scheduler.waiting), [waiting])

    def test_schedule_alternates_decode_and_waiting_prefill(self):
        scheduler = Scheduler(scheduler_config())
        running = Sequence([1, 2, 3])
        scheduler.block_manager.allocate(running, 0)
        running.status = SequenceStatus.RUNNING
        running.is_prefill = False
        scheduler.running.append(running)
        waiting = Sequence([4, 5, 6])
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertFalse(scheduler.waiting)
        self.assertEqual(list(scheduler.running), [running, waiting])

    def test_decode_schedule_rotates_running_sequences(self):
        scheduler = Scheduler(scheduler_config())
        scheduler.max_num_seqs = 2
        seqs = [Sequence([i, i + 1]) for i in range(3)]
        for seq in seqs:
            scheduler.block_manager.allocate(seq, 0)
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            scheduler.running.append(seq)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, seqs[:2])
        self.assertEqual(list(scheduler.running), [seqs[2], seqs[0], seqs[1]])


if __name__ == "__main__":
    unittest.main()
