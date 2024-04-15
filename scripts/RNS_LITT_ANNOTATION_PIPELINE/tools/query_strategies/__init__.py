from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence, LeastConfidenceRNS
from .margin_sampling import MarginSampling, MarginSamplingRNS
from .entropy_sampling import EntropySampling, EntropySamplingRNS
from .least_confidence_dropout import LeastConfidenceDropout,LeastConfidenceDropoutRNS
from .margin_sampling_dropout import MarginSamplingDropout, MarginSamplingDropoutRNS
from .entropy_sampling_dropout import EntropySamplingDropout,EntropySamplingDropoutRNS
from .kmeans_sampling import KMeansSampling, KMeansSamplingRNS
from .kcenter_greedy import KCenterGreedy, KCenterGreedyRNS
from .bayesian_active_learning_disagreement_dropout import BALDDropout, BALDDropoutRNS
from .kcenter_greedy_pca import KCenterGreedyPCA, KCenterGreedyPCARNS
from .mean_std import MeanSTD, MeanSTDRNS
from .waal import WAAL, WAALRNS
from .loss_prediction import LossPredictionLoss, LossPredictionLossRNS
from .badge_sampling import BadgeSampling, BadgeSamplingRNS

from .adversarial_bim import AdversarialBIM
from .adversarial_deepfool import AdversarialDeepFool
from .kmeans_sampling_gpu import KMeansSamplingGPU
from .var_ratio import VarRatio
from .ceal import CEALSampling
from .vaal import VAAL

