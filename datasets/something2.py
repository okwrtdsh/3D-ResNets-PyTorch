import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from multiprocessing import Pool
from itertools import chain
from functools import partial

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

# def video_loader(video_dir_path, frame_indices, image_loader):
#     # import pickle
#     import numpy as np
#     from glob import glob
#     cache_path = os.path.join(video_dir_path, 'cache.npy')
#     is_load = False
#
#     if os.path.isfile(cache_path):
#         try:
#             # with open(cache_path, 'rb') as f:
#             #     cache = pickle.load(f)
#             # frames = cache['frames']
#             frames = np.load(cache_path)
#             is_load = True
#         except Exception as e:
#             print(cache_path, e)
#
#     if not is_load:
#         # frames = []
#         # for image_path in sorted(glob(os.path.join(video_dir_path, '*.jpg'))):
#         #     frames.append(image_loader(image_path))
#         # cache = {
#         #     'frames': frames,
#         # }
#         # with open(cache_path, 'wb') as f:
#         #     pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
#         frames = []
#         for image_path in sorted(glob(os.path.join(video_dir_path, '*.jpg'))):
#             frames.append(np.array(image_loader(image_path)))
#         frames = np.array(frames).astype(np.uint8)
#         np.save(cache_path, frames)
#
#     video = []
#     for i in frame_indices:
#         # image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
#         # if os.path.exists(image_path):
#         #    video.append(image_loader(image_path))
#         if i <= len(frames):
#             # video.append(frames[i-1])
#             video.append(Image.fromarray(frames[i-1]))
#         else:
#             return video
#
#     return video


def video_loader2(video_dir_path, frame_indices, image_loader):
    import pickle
    from glob import glob
    cache_path = os.path.join(video_dir_path, 'cache.pkl')
    is_load = False

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            frames = cache['frames']
            is_load = True
        except Exception as e:
            print(cache_path, e)

    if not is_load:
        frames = []
        for image_path in sorted(glob(os.path.join(video_dir_path, '*.jpg'))):
            frames.append(image_loader(image_path))
        cache = {
            'frames': frames,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    video = []
    for i in frame_indices:
        if i <= len(frames):
            video.append(frames[i-1])
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader2, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_names.append(key)
            annotations.append(value['annotations'])

    return video_names, annotations


def _make_dataset(i, root_path, video_names, annotations, class_to_idx, n_samples_for_each_video, sample_duration):
    dataset = []
    if i % 1000 == 0:
        print('dataset loading [{}/{}]'.format(i, len(video_names)))

    video_path = os.path.join(root_path, video_names[i])
    if not os.path.exists(video_path):
        return dataset

    n_frames_file_path = os.path.join(video_path, 'n_frames')
    n_frames = int(load_value_file(n_frames_file_path))
    if n_frames <= 0:
        return dataset

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
        'video_id': video_names[i].split('/')[0]
    }
    if len(annotations) != 0:
        sample['label'] = class_to_idx[annotations[i]['label']]
    else:
        sample['label'] = -1

    if n_samples_for_each_video == 1:
        sample['frame_indices'] = list(range(1, n_frames + 1))
        dataset.append(sample)
    else:
        if n_samples_for_each_video > 1:
            step = max(1,
                       math.ceil((n_frames - 1 - sample_duration) /
                                 (n_samples_for_each_video - 1)))
        else:
            step = sample_duration
        for j in range(1, n_frames, step):
            sample_j = copy.deepcopy(sample)
            sample_j['frame_indices'] = list(
                range(j, min(n_frames + 1, j + sample_duration)))
            dataset.append(sample_j)
    return dataset


def make_dataset2(root_path, annotation_path, subset, n_samples_for_each_video,
                  sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    with Pool(os.cpu_count()) as p:
        res = p.map(
            partial(
                _make_dataset, root_path=root_path, video_names=video_names,
                annotations=annotations, class_to_idx=class_to_idx,
                n_samples_for_each_video=n_samples_for_each_video,
                sample_duration=sample_duration
            ),
            range(len(video_names))
        )
    dataset = list(chain(*res))
    return dataset, idx_to_class


def make_dataset3(root_path, annotation_path, subset, n_samples_for_each_video,
                  sample_duration):
    import pickle
    cache_path = os.path.join(root_path, '../cache', '%s-%s-%s-%s.pkl' % (
        os.path.basename(annotation_path).replace('.json', ''),
        subset,
        n_samples_for_each_video,
        sample_duration
    ))

    if os.path.isfile(cache_path):
        print('pickle load from', cache_path)
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        dataset = cache['dataset']
        idx_to_class = cache['idx_to_class']
    else:
        print('pickle dump to', cache_path)
        data = load_annotation_data(annotation_path)
        video_names, annotations = get_video_names_and_annotations(data, subset)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        with Pool(os.cpu_count()) as p:
            res = p.map(
                partial(
                    _make_dataset, root_path=root_path, video_names=video_names,
                    annotations=annotations, class_to_idx=class_to_idx,
                    n_samples_for_each_video=n_samples_for_each_video,
                    sample_duration=sample_duration
                ),
                range(len(video_names))
            )
        dataset = list(chain(*res))
        cache = {
            'dataset': dataset,
            'idx_to_class': idx_to_class,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset, idx_to_class


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[0]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class Something2(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 spatio_temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset3(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.spatio_temporal_transform = spatio_temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if self.spatio_temporal_transform is not None:
            clip = self.spatio_temporal_transform(clip)
        else:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
