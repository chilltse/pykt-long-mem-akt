"""Run from the repo root: python -m examples.fmkc_inference [--student demo-1].

Uses FMKC_CHECKPOINT_PATH and KT_STATE_DB_PATH; records real demo events only for
this demo student. Never run against a production student's ID.
"""
import argparse
import json
import uuid

from pykt.serving import AKTFMKCInferenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', default='demo-' + uuid.uuid4().hex)
    args = parser.parse_args()
    engine = AKTFMKCInferenceEngine.from_env()
    try:
        # Decode a known training identity so the example works for any field count.
        questions = engine.encoder.encoding['questions']
        tokens = json.loads(next(iter(questions))) if questions else [None] * len(engine.encoder.field_names)
        vector = [None if token is None else json.loads(token) for token in tokens]
        candidates = [dict(candidate_id='next-onset', kc_vector=vector)]
        print('model', json.dumps(engine.model_info(), ensure_ascii=False))
        print('before', engine.predict_candidates(args.student, candidates))
        revision = engine.get_student_state(args.student)['history_revision']
        attempt_id = uuid.uuid4().hex
        events = [dict(event_id=f'{attempt_id}:{i}', kc_vector=vector, response=i % 2) for i in range(3)]
        print('real observations', engine.observe_batch(args.student, events, revision, attempt_id))
        print('idempotent retry', engine.observe_batch(args.student, events, revision, attempt_id))
        branch = engine.fork_student(args.student)
        for response in (1, 0, 1):
            print('hypothesis', engine.simulate_step(branch['branch_id'], vector, response))
        print('branch forecast', engine.predict_branch(branch['branch_id'], candidates))
        engine.discard_branch(branch['branch_id'])
        print('forced rollout', engine.run_rollout(args.student,
            [dict(kc_vector=vector, response=1)] * 3, mode='forced', seed=42,
            forecast_candidates=candidates))
        print('Monte Carlo', engine.run_rollout(args.student, [dict(kc_vector=vector)] * 3,
            mode='sample', seed=42, num_trajectories=4, forecast_candidates=candidates))
        print('real state after simulations', engine.get_student_state(args.student))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
