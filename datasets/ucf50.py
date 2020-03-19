import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np


class UCF50(data.Dataset):
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
                 n_samples_for_each_video=2,
                 spatial_transform=None,
                 temporal_transform=None,
                 spatio_temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 test_split=1):

        self.root_path = root_path
        self.data = self.make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, test_split)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.spatio_temporal_transform = spatio_temporal_transform
        self.target_transform = target_transform
        self._videos = {}
        self.class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing',
            'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Lunges',
            'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse',
            'Pullup', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
            'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']

    def make_dataset(self, root_path, annotation_path, subset, n_samples_for_each_video, sample_duration, test_split):
        with open(annotation_path, 'r') as f:
            samples = json.load(f)

        data = []
        for sample in samples:
            if subset == 'training':
                if test_split == int(sample['group']):
                    continue
            elif test_split != int(sample['group']):
                continue

            n_frames = sample['n_frames']
            if n_samples_for_each_video == 1:
                sample['frame_indices'] = list(range(1, n_frames + 1))
                data.append(sample)
            else:
                if n_samples_for_each_video > 1:
                    step = max(1, math.ceil((n_frames - 1 - sample_duration) / (n_samples_for_each_video - 1)))
                else:
                    step = sample_duration
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + sample_duration)))
                    data.append(sample_j)
        return data


    def get_video(self, path):
        if path in self._videos:
            return self._videos[path]
        video = np.load(os.path.join(self.root_path, path))
        self._videos[path] = video
        return video


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        video = self.get_video(path)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(self.data[index]['frame_indices'])

        clip = []
        for i in frame_indices:
            if i <= video.shape[0]:
                clip.append(Image.fromarray(video[i-1]))
            else:
                clip.append(Image.fromarray(video[-1]))

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

