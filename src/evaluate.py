

from pathlib import Path
import trackeval


project_dir = Path(__file__).resolve().parent.parent

eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config["USE_PARALLEL"] = False
eval_config["PLOT_CURVES"] = False

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

dataset_config["GT_FOLDER"] = str(project_dir / "data")
dataset_config["TRACKERS_FOLDER"] = str(project_dir / "results")

dataset_config["TRACKERS_TO_EVAL"] = ["iou"]
dataset_config["SEQ_INFO"] = {"MOT17-05-FRCNN": None}

dataset_config["SKIP_SPLIT_FOL"] = True
dataset_config["TRACKER_SUB_FOLDER"] = ""

evaluator = trackeval.Evaluator(eval_config)

dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

metrics = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR(),
    trackeval.metrics.Identity()
]

evaluator.evaluate([dataset], metrics)
