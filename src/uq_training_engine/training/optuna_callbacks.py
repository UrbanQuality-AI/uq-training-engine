import logging

from optuna.study import Study
from optuna.trial import FrozenTrial

logger = logging.getLogger(__name__)


def optuna_logging_callback(study: Study, trial: FrozenTrial) -> None:
    """
    Log trial completion, objective value, parameters, and running best after each trial.

    :param study: The Optuna ``Study`` instance.
    :param trial: The ``FrozenTrial`` that just finished.
    :return: None.

    Example::
        In: optuna_logging_callback(study, trial)
        Out: logs e.g. "[Optuna] Trial 3 finished." and best value
    """
    logger.info("[Optuna] Trial %d finished.", trial.number)
    logger.info("[Optuna] Objective value: %.4f | Parameters: %s", trial.value, trial.params)
    logger.info("[Optuna] Best value so far: %.4f", study.best_value)
