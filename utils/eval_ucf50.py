import json

import numpy as np
import pandas as pd

class UCFclassification(object):

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 subset='validation', verbose=False, top_k=1, test_split=None):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.top_k = top_k
        self.test_split = test_split
        self.ap = None
        self.hit_at_k = None
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        # if not all([field in data.keys() for field in self.gt_fields]):
            # raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        activity_index = {
            'BaseballPitch': 0,
            'Basketball': 1,
            'BenchPress': 2,
            'Biking': 3,
            'Billards': 4,
            'BreastStroke': 5,
            'CleanAndJerk': 6,
            'Diving': 7,
            'Drumming': 8,
            'Fencing': 9,
            'GolfSwing': 10,
            'HighJump': 11,
            'HorseRace': 12,
            'HorseRiding': 13,
            'HulaHoop': 14,
            'JavelinThrow': 15,
            'JugglingBalls': 16,
            'JumpRope': 17,
            'JumpingJack': 18,
            'Kayaking': 19,
            'Lunges': 20,
            'MilitaryParade': 21,
            'Mixing': 22,
            'Nunchucks': 23,
            'PizzaTossing': 24,
            'PlayingGuitar': 25,
            'PlayingPiano': 26,
            'PlayingTabla': 27,
            'PlayingViolin': 28,
            'PoleVault': 29,
            'PommelHorse': 30,
            'Pullup': 31,
            'Punch': 32,
            'PushUps': 33,
            'RockClimbingIndoor': 34,
            'RopeClimbing': 35,
            'Rowing': 36,
            'SalsaSpin': 37,
            'SkateBoarding': 38,
            'Skiing': 39,
            'Skijet': 40,
            'SoccerJuggling': 41,
            'Swing': 42,
            'TaiChi': 43,
            'TennisSwing': 44,
            'ThrowDiscus': 45,
            'TrampolineJumping': 46,
            'VolleyballSpiking': 47,
            'WalkingWithDog': 48,
            'YoYo': 49}
        video_lst, label_lst = [], []
        """
        sample = {
            'video': os.path.join(group, action, clip)+'.npy',
            'n_frames': n_frames,
            'group': group,
            'label': labels.index(action),
            'action': action,
            'video_id': clip,
        }
        """
        for sample in data:
            if self.test_split != int(sample['group']):
                continue
            video_lst.append(sample['video_id'])
            label_lst.append(sample['label'])
        ground_truth = pd.DataFrame({'video-id': video_lst, 'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        # if not all([field in data.keys() for field in self.pred_fields]):
            # raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        video_lst, label_lst, score_lst = [], [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        if self.verbose:
            print(('[RESULTS] Performance on ActivityNet untrimmed video '
                   'classification task.'))
            print('\tError@{}: {}'.format(self.top_k, 1.0 - hit_at_k))
            #print '\tAvg Hit@{}: {}'.format(self.top_k, avg_hit_at_k)
        self.hit_at_k = hit_at_k

################################################################################
# Metrics
################################################################################
def compute_video_hit_at_k(ground_truth, prediction, top_k=3):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_vid[i] = np.mean([1 if this_label in pred_label else 0
                                       for this_label in gt_label])
    return float(avg_hits_per_vid.mean())


if __name__ == '__main__':
    #pdb.set_trace()
    import sys
    data_path = sys.argv[1]
    result_path = sys.argv[2]
    top_k = int(sys.argv[3])
    test_split = int(sys.argv[4])

    ucf_classification = UCFclassification(data_path, result_path, subset='validation', top_k=top_k, test_split=test_split)
    ucf_classification.evaluate()
    print(ucf_classification.hit_at_k)

