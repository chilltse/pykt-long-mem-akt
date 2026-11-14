import json
import pytest
from pianokt_backend.experiments.local_features import load_local_features,NAMES


def test_local_input_preserves_types_and_orders(tmp_path):
    root=tmp_path/'features';root.mkdir();(root/'segment').mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(kc_labeler_version='1.2.0',kc_feature_names=NAMES)))
    artifact=dict(artifact_type='pianokt_recommendation_kc',kc_labeler_version='1.2.0',kc_feature_names=NAMES,
        song_id='song',segment_id='segment',segment_index=1,midi_sha256='a'*64,reference_events_count=2,
        onsets=[dict(onset_time=i,kc_vector=[False,1,False,None,1,None]) for i in [0,1]])
    (root/'segment'/'artifact.json').write_text(json.dumps(artifact))
    def attempt(t,y): return dict(ordering_timestamp=t,attempt_id=t,kc_labeler_version='1.2.0',
        events_data=dict(kc_vectors=[[False,1,False,None,1,None]],correct=[y],onset_seconds=[0]))
    history=tmp_path/'history.json';history.write_text(json.dumps(dict(source='synthetic',students=[dict(learner_key='x',
        visits=[dict(attempts=[attempt('2026-01-02T00:00:00+00:00',0),attempt('2026-01-01T00:00:00+00:00',1)])])])))
    data=load_local_features(root,history)
    assert data['history_origin']=='synthetic'
    assert [e['response'] for e in data['students'][0]['history']]==[1,0]
    assert data['candidates'][0]['vectors'][0][0] is False
    assert data['candidates'][0]['vectors'][0][3] is None
    assert data['ready_status_checked'] is False
    artifact['kc_feature_names']=NAMES[::-1]
    (root/'segment'/'artifact.json').write_text(json.dumps(artifact))
    with pytest.raises(ValueError):load_local_features(root,history)
