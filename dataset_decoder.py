from datasets.ucf101_decoder import UCF101


def get_training_set(opt,
                     common_temporal_transform,
                     common_spatial_transform,
                     target_spatial_transform,
                     input_spatial_transform,
                     target_label_transform
                     ):
    assert opt.dataset in [
        'ucf101',
    ]

    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            common_temporal_transform=common_temporal_transform,
            common_spatial_transform=common_spatial_transform,
            target_spatial_transform=target_spatial_transform,
            input_spatial_transform=input_spatial_transform,
            target_label_transform=target_label_transform,
            sample_duration=opt.sample_duration)
    return training_data


def get_validation_set(opt,
                       common_temporal_transform,
                       common_spatial_transform,
                       target_spatial_transform,
                       input_spatial_transform,
                       target_label_transform
                       ):
    assert opt.dataset in [
        'ucf101',
    ]

    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            common_temporal_transform=common_temporal_transform,
            common_spatial_transform=common_spatial_transform,
            target_spatial_transform=target_spatial_transform,
            input_spatial_transform=input_spatial_transform,
            target_label_transform=target_label_transform,
            sample_duration=opt.sample_duration)

    return validation_data


def get_test_set(opt,
                 common_temporal_transform,
                 common_spatial_transform,
                 target_spatial_transform,
                 input_spatial_transform,
                 target_label_transform
                 ):
    assert opt.dataset in [
        'ucf101',
    ]
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            10,
            common_temporal_transform=common_temporal_transform,
            common_spatial_transform=common_spatial_transform,
            target_spatial_transform=target_spatial_transform,
            input_spatial_transform=input_spatial_transform,
            target_label_transform=target_label_transform,
            sample_duration=opt.sample_duration)

    return test_data
