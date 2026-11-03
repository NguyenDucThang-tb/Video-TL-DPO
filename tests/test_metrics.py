from video_tldpo.evaluation.metrics import question_metrics
def test_transitions():
    ref=[{"sample_id":"a","correct":False},{"sample_id":"b","correct":True}]
    cur=[{"sample_id":"a","correct":True,"prediction":"yes"},{"sample_id":"b","correct":False,"prediction":"no"}]
    x=question_metrics(cur,ref); assert x["wrong_to_correct"]==1 and x["correct_to_wrong"]==1 and x["delta_accuracy_pp"]==0
