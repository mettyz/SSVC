from .r2d3d import *
from .r3d import *
from .s3dg import S3D


def select_model(network, first_channel=3):
    """
    Support Networks:
        23d_resnet18
        23d_resnet34
        23d_resnet50
        s3d
        s3dg
        3d_resnet18
        3d_resnet50
    """
    param = {'feature_size': 1024}
    if network == 'r23d18':
        model = resnet18_2d3d_full(track_running_stats=True)
        param['feature_size'] = 256
    elif network == 'r23d34':
        model = resnet34_2d3d_full(track_running_stats=True)
        param['feature_size'] = 256 
    elif network == 'r23d50':
        model = resnet50_2d3d_full(track_running_stats=True)
    elif network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r3d18':
        model = resnet18()
        param['feature_size'] = model.out_feature_size
    elif network == 'r3d50':
        model = resnet50()
        param['feature_size'] = model.out_feature_size
    else: 
        raise NotImplementedError

    return model, param
