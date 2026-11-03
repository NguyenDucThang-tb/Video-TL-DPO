import numpy as np
from video_tldpo.video.sampling import uniform_indices
from video_tldpo.video.temporal_masking import mask_temporal
from video_tldpo.video.spatial_masking import mask_boxes

def test_sampling_and_temporal_masking():
    assert uniform_indices(32,8)==[0,4,9,13,18,22,27,31]
    x=np.ones((4,4,4,3),dtype=np.uint8)*100; y=mask_temporal(x,[1,3],mode="zero")
    assert y.shape==x.shape and np.all(y[[0,2]]==x[[0,2]]) and np.all(y[[1,3]]==0)

def test_box_masking():
    x=np.ones((2,5,6,3),dtype=np.uint8); y=mask_boxes(x,{0:[[1,1,3,3]]})
    assert np.all(y[0,1:3,1:3]==0) and np.all(y[1]==1)
