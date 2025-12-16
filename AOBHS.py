import random
from copy import deepcopy
import numpy as np
import itertools
from sklearn.neighbors import NearestNeighbors
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTreeClassifier
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.utils import get_dimensions, normalize_values_in_dict, check_random_state
from collections import deque
import math


def get_all_neighbors(X, K=5):
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(X)
    return knn


def get_minority_neighbors(minority_samples, K=5):
    knn = NearestNeighbors(n_neighbors=min(K, len(minority_samples)))
    knn.fit(minority_samples)
    return knn


class AOBHSClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self,
                 n_estimators=10,
                 max_features='auto',
                 lambda_value=6,
                 performance_metric='acc',
                 drift_detection_method=ADWIN(0.001),
                 warning_detection_method=ADWIN(0.01),
                 window_size=200,
                 rule_of_lambda='6',
                 random_state=None,
                 classSize=None,
                 feature_selection='random_n',
                 k_value = 5,
                 theta_value = 0.9):
        self.theta = theta_value
        self.k_value = k_value
        if performance_metric in ['acc', 'kappa']:
            self.performance_metric = performance_metric
        else:
            raise ValueError('Invalid performance metric: {}'.format(performance_metric))
        self.disable_weighted_vote = None
        self.classSize = classSize
        self._train_weight_seen_by_model = None
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method
        self.window_size = window_size
        self.rule_of_lambda = rule_of_lambda
        self.random_state = random_state
        self._random_state = np.random.RandomState(random_state)
        self.batch_window = deque(maxlen=self.window_size)
        self.window_classifier_weight = [deque(maxlen=self.window_size) for _ in range(self.n_estimators)]
        self.drift_window = deque(maxlen=self.window_size)
        self.sketch = {}
        self.ensemble = None
        self.instances_seen = 0
        self.classes = None
        self.classRatio = 1
        self.imbalanceThreshold = 5
        if feature_selection not in ['none', 'first_n', 'random_n']:
            raise ValueError('Invalid feature selection method: {}'.format(feature_selection))
        self.feature_selection = feature_selection

    def _init_ensemble(self, X):
        self._set_max_features(get_dimensions(X)[1])
        self.ensemble = []
        for i in range(self.n_estimators):
            feature_selection = 'first_n' if i == 0 else 'random_n'
            self.ensemble.append(AOBHSBaseLearner(index_original=i,
                                                 classifier=ARFHoeffdingTreeClassifier(),
                                                 drift_detection_method=self.drift_detection_method,
                                                 warning_detection_method=self.warning_detection_method,
                                                 batch_window=self.batch_window,
                                                 window_classifier_weight=self.window_classifier_weight[i],
                                                 instances_seen=self.instances_seen,
                                                 is_background_learner=False,
                                                 feature_selection=feature_selection))

    def _set_max_features(self, n):
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = round(math.sqrt(n))
        elif self.max_features == 'log2':
            self.max_features = round(math.log2(n))
        elif isinstance(self.max_features, int):
            pass
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n)
        elif self.max_features is None:
            self.max_features = n
        else:
            self.max_features = round(math.sqrt(n))
        if self.max_features < 0:
            self.max_features += n
        if self.max_features <= 0:
            self.max_features = 1
        if self.max_features > n:
            self.max_features = n

    def dynamic_tau(self):
        if any(self.drift_window):
            return 0.1
        else:
            return 0.5

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.classes is None and classes is not None:
            self.classes = classes
        self.instances_seen += X.shape[0]
        print(str(self.instances_seen))
        if X.ndim == 1:
            X = X.reshape(1, -1)
            y = np.array([y])
        for i in range(X.shape[0]):
            self.batch_window.append(np.hstack((X[i], y[i])))
        if (self.instances_seen > self.window_size) and (self.ensemble is None):
            self._init_ensemble(X)
        self._update_sketch(X, y)
        self.update_class_size(y, classes)
        self.onlineBaggingTrain(X, y, isSynthetic=False)
        if self.classRatio > self.imbalanceThreshold:
            current_minority_class = self.get_minority_class()
            synthetic_samples = self.generate_synthetic_sample(current_minority_class, K=self.k_value)
            if synthetic_samples is None:
                return
            synthetic_samples = synthetic_samples.reshape(1, -1)
            self.update_class_size(np.array([self.get_minority_class()]), classes)
            self.onlineBaggingTrain(synthetic_samples, np.array([self.get_minority_class()]), isSynthetic=True)

    def onlineBaggingTrain(self, X, y, isSynthetic):
        for i in range(self.n_estimators):
            for j in range(X.shape[0]):
                lambda_value = self.calculate_poisson_lambda(y,self.lambda_value)
                k = self._random_state.poisson(lambda_value)
                if isSynthetic is False:
                    self.window_classifier_weight[i].append(k)
                if (k > 0) and (self.ensemble is not None):
                    drift_status = self.ensemble[i].partial_fit(np.array(X[j]), np.array(y[j]),
                                                                classes=self.classes,
                                                                sample_weight=np.array([k]),
                                                                instances_seen=self.instances_seen,
                                                                minority_class=self.get_minority_class(),
                                                                classRatio=self.classRatio)
                    self.drift_window.append(drift_status)

    def _update_sketch(self, X, y):
        for idx in range(len(np.array(X))):
            current_X = X[idx]
            current_y = y[idx]
            if current_y not in self.sketch:
                self.sketch[current_y] = {
                    'count': 0,
                    'mean': np.zeros_like(current_X),
                    'cov': np.zeros((current_X.shape[0], current_X.shape[0]))
                }
            self.sketch[current_y]['count'] += 1
            alpha = 0.01
            self.sketch[current_y]['mean'] = (1 - alpha) * self.sketch[current_y]['mean'] + alpha * current_X
            delta = current_X - self.sketch[current_y]['mean']
            self.sketch[current_y]['cov'] += np.outer(delta, delta) / max(1, self.sketch[current_y]['count'])

    def generate_synthetic_sample(self, target_class, K=5):
        window = np.squeeze(np.array(self.batch_window))
        if len(self.batch_window) == 0:
            return None
        all_samples = np.array([row[:-1] for row in window])
        minority_samples = np.array([row[:-1] for row in window if row[-1] == target_class])
        if len(minority_samples) < 2:
            return None
        tau = self.dynamic_tau()
        all_knn = get_all_neighbors(all_samples, K)
        min_knn = get_minority_neighbors(minority_samples, K)
        for idx, sample in enumerate(minority_samples):
            neighbors_idx = all_knn.kneighbors([sample], return_distance=False)[0]
            neighbor_labels = [window[i][-1] for i in neighbors_idx]
            minority_count = sum(1 for lbl in neighbor_labels if lbl == target_class)
            minority_ratio = minority_count / K
            if minority_ratio >= tau:
                min_neighbors_idx = min_knn.kneighbors([sample], return_distance=False)[0]
                neighbor_sample = minority_samples[random.choice(min_neighbors_idx[1:])]
                new_sample = sample + np.random.rand() * (neighbor_sample - sample)
                if target_class in self.sketch:
                    mean = self.sketch[target_class]['mean']
                    cov = self.sketch[target_class]['cov']
                    adjustment = np.random.multivariate_normal(mean, cov * 0.1)
                    new_sample = (new_sample + adjustment) / 2
                return new_sample
        return None

    def update_class_size(self, y, classes):
        if self.classSize is None:
            num_classes = len(classes)
            self.classSize = np.ones(num_classes) / num_classes
        for j in range(len(np.array(y))):
            self.classSize = self.theta * self.classSize + (1 - self.theta) * (
                    y[j] == np.arange(len(self.classSize)))
        self.update_class_ratio()

    def update_class_ratio(self):
        self.classRatio = max(self.classSize) / (min(self.classSize) + 1e-6)

    def calculate_poisson_lambda(self, y, lambda_value=6):
        min_class = self.get_minority_class()
        y = np.array(y)
        has_drift = self.drift_detection_method.detected_change()
        if y.size == 1:
            if y.item() == min_class:
                return lambda_value * math.sqrt(self.classRatio) if has_drift else math.sqrt(self.classRatio)
            else:
                return lambda_value/math.sqrt(self.classRatio) if has_drift else 1/math.sqrt(self.classRatio)
        else:
            if np.any(y == min_class):
                return lambda_value * math.sqrt(self.classRatio) if has_drift else math.sqrt(self.classRatio)
            else:
                return lambda_value/math.sqrt(self.classRatio) if has_drift else 1/math.sqrt(self.classRatio)

    def get_majority_class(self):
        return np.argmax(self.classSize)

    def get_minority_class(self):
        return np.argmin(self.classSize)

    def predict(self, X):
        if self.ensemble is None:
            self._init_ensemble(X)
        y_proba = self.predict_proba(X)
        n_rows = y_proba.shape[0]
        y_pred = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            index = np.argmax(y_proba[i])
            y_pred[i] = index
        return y_pred

    def predict_proba(self, X):
        if self.ensemble is None:
            self._init_ensemble(X)
        r, _ = get_dimensions(X)
        y_proba = []
        for i in range(r):
            votes = deepcopy(self._get_votes_for_instance(X[i]))
            if votes == {}:
                y_proba.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes)
                if self.classes is not None:
                    votes_array = np.zeros(int(max(self.classes)) + 1)
                else:
                    votes_array = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    votes_array[int(key)] = value
                y_proba.append(votes_array)
        if self.classes is not None:
            y_proba = np.asarray(y_proba)
        else:
            y_proba = np.asarray(list(itertools.zip_longest(*y_proba, fillvalue=0.0))).T
        return y_proba

    def reset(self):
        self.ensemble = None
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self._random_state = check_random_state(self.random_state)
        self.batch_window = deque(maxlen=self.window_size)
        self.window_classifier_weight = [deque(maxlen=self.window_size) for i in range(self.n_estimators)]

    def _get_votes_for_instance(self, X):
        if self.ensemble is None:
            self._init_ensemble(X)
        combined_votes = {}
        for i in range(self.n_estimators):
            vote = deepcopy(self.ensemble[i]._get_votes_for_instance(X))
            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=True)
                if not self.disable_weighted_vote:
                    performance = self.ensemble[i].weight \
                        if self.performance_metric == 'acc' \
                        else self.ensemble[i].evaluator.kappa_score()
                    if performance != 0.0:
                        for k in vote:
                            vote[k] = vote[k] * performance
                for k in vote:
                    try:
                        combined_votes[k] += vote[k]
                    except KeyError:
                        combined_votes[k] = vote[k]
        return combined_votes


class AOBHSBaseLearner(BaseSKMObject):
    def __init__(self,
                 index_original,
                 classifier: ARFHoeffdingTreeClassifier,
                 instances_seen,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 is_background_learner,
                 batch_window,
                 window_classifier_weight,
                 feature_selection='random_n',
                 beta=0.5,
                 theta=0.01,
                 period=50,
                 performance_threshold=0.7,
                 replacement_threshold=0.8):
        self.index_original = index_original
        self.classifier = classifier
        self.created_on = instances_seen
        self.is_background_learner = is_background_learner
        self.evaluator_method = ClassificationPerformanceEvaluator
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method
        self.last_drift_on = 0
        self.last_warning_on = 0
        self.nb_drifts_detected = 0
        self.nb_warnings_detected = 0
        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        self._use_drift_detector = False
        self._use_background_learner = False
        self.evaluator = self.evaluator_method()
        self.drift_window = deque(maxlen=len(batch_window))
        self.feature_selection = feature_selection
        self.beta = beta
        self.theta = theta
        self.period = period
        self.epochs = 0
        self.weight = 1.0
        self.background_weight = 1.0
        self.background_performance_history = deque(maxlen=period)
        self.performance_history = deque(maxlen=period)
        self.current_performance = 0.0
        self.background_performance = 0.0
        self.performance_threshold = performance_threshold
        self.replacement_threshold = replacement_threshold
        if drift_detection_method is not None:
            self._use_drift_detector = True
            self.drift_detection = deepcopy(drift_detection_method)
        if warning_detection_method is not None:
            self._use_background_learner = True
            self.warning_detection = deepcopy(warning_detection_method)
        self.latest_window = batch_window
        self.latest_weight = window_classifier_weight
        if len(self.latest_window) != 0:
            self.latest_x = np.array(self.latest_window)[:, :-1]
            self.latest_y = np.array(self.latest_window)[:, -1]
            self.classifier.partial_fit(self.latest_x, self.latest_y,
                                        sample_weight=np.array(self.latest_weight))

    def _should_create_background_learner(self, current_performance):
        if self.is_background_learner:
            return False
        return (self.weight < 1) or \
            (self._use_background_learner and self.warning_detection.detected_change())

    def _should_replace_classifier(self, current_performance):
        if self.background_learner is None:
            return False
        return (self.weight * 10 < self.background_weight) or \
            (self._use_drift_detector and self.drift_detection.detected_change())

    def _replace_with_background_learner(self):
        if self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.weight = self.background_weight
            self.performance_history = self.background_performance_history.copy()
            self.current_performance = self.background_performance
            self.background_learner = None
            self.background_weight = 1.0
            self.background_performance_history = deque(maxlen=self.period)
            self.background_performance = 0.0

    def partial_fit(self, x, y, classes, sample_weight, instances_seen, minority_class=None, classRatio=None):
        self.epochs += 1
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)
        if isinstance(sample_weight, int):
            sample_weight = np.array([sample_weight])
        drift_status = self.classifier.partial_fit(x, y, classes=classes, sample_weight=sample_weight)
        self.drift_window.append(drift_status)
        y_pred = self.classifier.predict(x)
        is_correct = (y_pred == y).astype(int)
        self.performance_history.append(is_correct[0])
        self.current_performance = np.mean(self.performance_history)
        if self.background_learner is not None:
            backup_pred = self.background_learner.predict(x)
            backup_is_correct = (backup_pred == y).astype(int)
            self.background_performance_history.append(backup_is_correct[0])
            self.background_performance = np.mean(self.background_performance_history)
        if self._should_create_background_learner(self.current_performance):
            background_learner = self.classifier.new_instance()
            self.background_learner = AOBHSBaseLearner(
                self.index_original,
                background_learner,
                self.created_on,
                self.drift_detection_method,
                self.warning_detection_method,
                True,
                batch_window=self.latest_window,
                window_classifier_weight=self.latest_weight,
                feature_selection=self.feature_selection
            )
            self.background_learner.partial_fit(x, y, classes=classes,
                                                sample_weight=sample_weight,
                                                instances_seen=instances_seen,
                                                minority_class=minority_class,
                                                classRatio=classRatio)
            self.background_performance_history = deque(maxlen=self.period)
            self.background_performance = 0.0
        if self._should_replace_classifier(self.current_performance):
            self._replace_with_background_learner()
        new_weight = (1 / classRatio) if (minority_class != int(y)) else 1
        if is_correct[0]:
            self.weight = self.weight * self.beta + 1 + new_weight
        else:
            self.weight = self.weight * self.beta + 1 - new_weight
        if self.background_learner is not None:
            backup_pred = self.background_learner.predict(x)
            backup_is_correct = (backup_pred == int(y)).astype(int)
            backup_new_weight = (1 / classRatio) if (minority_class != y) else 1
            if backup_is_correct[0]:
                self.background_weight = self.background_weight * self.beta + 1 + backup_new_weight
            else:
                self.background_weight = self.background_weight * self.beta + 1 - backup_new_weight
        if self._use_drift_detector and not self.is_background_learner:
            if self._use_background_learner:
                self.warning_detection.add_element(int(not is_correct[0]))
                if self.warning_detection.detected_change():
                    print("AOBHS Drift warning at " + str(instances_seen))
                    self.last_warning_on = instances_seen
                    self.nb_warnings_detected += 1
                    background_learner = self.classifier.new_instance()
                    self.background_learner = AOBHSBaseLearner(
                        self.index_original,
                        background_learner,
                        self.created_on,
                        self.drift_detection_method,
                        self.warning_detection_method,
                        True,
                        batch_window=self.latest_window,
                        window_classifier_weight=self.latest_weight,
                        feature_selection=self.feature_selection
                    )
                    self.background_learner.partial_fit(x, y, classes=classes,
                                                        sample_weight=sample_weight,
                                                        instances_seen=instances_seen,
                                                        minority_class=minority_class,
                                                        classRatio=classRatio,
                                                        )
            self.drift_detection.add_element(int(not is_correct[0]))
            if self.drift_detection.detected_change():
                print("AOBHS Drift detected at " + str(instances_seen))
                self.last_drift_on = instances_seen
                self.nb_drifts_detected += 1
                self.reset(instances_seen)
                return True
        return False

    def predict(self, x):
        return self.classifier.predict(x)

    def _get_votes_for_instance(self, x):
        return self.classifier._get_votes_for_instance(x)

    def reset(self, instances_seen):
        if self._use_background_learner and self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.evaluator_method = self.background_learner.evaluator_method
            self.created_on = self.background_learner.created_on
            self.feature_selection = self.background_learner.feature_selection
            self.background_learner = None
        else:
            self.classifier.reset()
            self.created_on = instances_seen
            self.drift_detection.reset()
            self.latest_x = np.array(self.latest_window)[:, :-1]
            self.latest_y = np.array(self.latest_window)[:, -1]
            self.classifier.partial_fit(self.latest_x, self.latest_y,
                                        sample_weight=np.array(self.latest_weight))
        self.evaluator = self.evaluator_method()
