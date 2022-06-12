# -*- coding: UTF-8 -*-

import contextlib
import copy
import io
import itertools
import json
import logging
import sys

import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from collections import OrderedDict, defaultdict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

from .parsing_eval_api import ParsingEval
from .parsing_eval_api import ParsingGT
from .parsing_eval_api import get_ann_fields


class ParsingEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            tasks='parsing',
            distributed=True,
            output_dir=None,
            *,
            max_dets_per_image=None,
            parsing_metrics=('mIoU', 'APp', 'APr', 'APh'),
    ):
        """

        :param dataset_name:
        :param tasks:
        :param distributed:
        :param output_dir:
        :param max_dets_per_image:
        :param parsing_metrics:
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]

        self._max_dets_per_image = max_dets_per_image
        self._tasks = tasks
        self.metrics = parsing_metrics

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
            ann_fields = defaultdict(dict, get_ann_fields(dataset_name))
            ann_fields['parsing'].update({'semseg_format': "mask"})
            ann_fields = dict(ann_fields)
            self.parsing_GT = ParsingGT(self._metadata.image_root, self._metadata.json_file, set('parsing'), ann_fields)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = True  # "annotations" in self._coco_api.dataset

    def reset(self):
        self._semseg_predictions = []
        self._part_predictions   = []
        self._human_predictions  = []
        self._parsing_predictions   = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: [{"parsing": {
                        "semseg_outputs": ndarray(H, W)
                        "parsing_outputs": [{
                                            "category_id": 1,
                                            "parsing": ndarray(H, W),
                                            "instance_score": float,
                                            "parsing_bbox_score": float,
                                            "part_pixel_scores": [float, ...]
                                            } ... ],

                        "part_outputs": [{
                                            "category_id": int,
                                            "score": float,
                                            "mask": ndarray(H, W),
                                         } ... ],
                        "human_outputs": [{
                                            "category_id": 0,
                                            "score": float,
                                            "mask": ndarray(H, W),
                                          } ... ],
                     }}, ...]
            During evaluation, length of outputs is 1
        """

        output_dict = outputs[-1]['parsing']

        self._semseg_predictions.append(
            {inputs[0]["file_name"].split('/')[-1]: csr_matrix(output_dict["semseg_outputs"].argmax(dim=0).cpu().numpy())}
        )

        for parsing_output in output_dict["parsing_outputs"]:
            parsing_prediction = {"image_id": inputs[0]["image_id"]}
            # parsing_prediction['category_id'] = parsing_output["category_id"]
            parsing_prediction['parsing'] = csr_matrix(parsing_output["parsing"].numpy())
            parsing_prediction['score'] = parsing_output["instance_score"]
            if len(parsing_prediction) > 1:
                self._parsing_predictions.append(parsing_prediction)

        for part_output in output_dict["part_outputs"]:
            part_prediction = {"image_id": inputs[0]["image_id"]}
            part_prediction["img_name"] = inputs[0]["file_name"].split('/')[-1].split('.')[0]
            part_prediction["category_id"] = part_output["category_id"]
            part_prediction["score"] = part_output["score"]
            part_prediction["mask"] = csr_matrix(np.array(part_output["mask"] > 0).astype(np.uint8))
            if len(part_prediction) > 1:
                self._part_predictions.append(part_prediction)

        for human_output in output_dict["human_outputs"]:
            human_prediction = {"image_id": inputs[0]["image_id"]}
            human_prediction["img_name"] = inputs[0]["file_name"].split('/')[-1].split('.')[0]
            human_prediction["category_id"] = human_output["category_id"]
            human_prediction["score"] = human_output["score"]
            human_prediction["mask"] = csr_matrix(np.array(human_output["mask"] > 0).astype(np.uint8))
            if len(human_prediction) > 1:
                self._human_predictions.append(human_prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """


        if self._distributed:
            self._logger.info("gathering results from all devices....")
            comm.synchronize()

            semseg_predictions = comm.gather(self._semseg_predictions, dst=0)
            semseg_predictions = list(itertools.chain(*semseg_predictions))

            part_predictions   = comm.gather(self._part_predictions, dst=0)
            part_predictions   = list(itertools.chain(*part_predictions))

            human_predictions  = comm.gather(self._human_predictions, dst=0)
            human_predictions  = list(itertools.chain(*human_predictions))

            parsing_prediction   = comm.gather(self._parsing_predictions, dst=0)
            parsing_prediction   = list(itertools.chain(*parsing_prediction))

            if not comm.is_main_process():
                return {}
        else:
            self._logger.info("gathering results from single devices....")

            semseg_predictions = self._semseg_predictions
            part_predictions   = self._part_predictions
            human_predictions  = self._human_predictions
            parsing_prediction   = self._parsing_predictions

        self._logger.info("gather results from all devices done")

        if len(part_predictions) == 0:
            self._logger.warning("[ParsingEvaluator] Did not receive valid part predictions.")
        if len(human_predictions) == 0:
            self._logger.warning("[ParsingEvaluator] Did not receive valid human predictions.")
        if len(parsing_prediction) == 0:
            self._logger.warning("[ParsingEvaluator] Did not receive valid parsing predictions.")

        self._results = OrderedDict()

        self._eval_parsing_predictions(semseg_predictions, parsing_prediction, part_predictions, human_predictions,)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_parsing_predictions(self, semseg_predictions, pars_predictions, part_predictions, human_predictions, img_ids=None):
        """
        Evaluate parsing predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        semseg_results  = semseg_predictions
        parsing_results = pars_predictions
        part_results    = part_predictions
        human_results   = human_predictions

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        _evaluate_predictions_on_parsing(
            self.parsing_GT,
            semseg_results,
            parsing_results,
            part_results,
            human_results,
            self._metadata,
            self._output_dir,
            self.metrics,
        )

def _evaluate_predictions_on_parsing(
        parsing_gt,
        semseg_results,
        parsing_results,
        part_results,
        human_results,
        metadata,
        output_folder,
        metrics,
):
    """
    Evaluate the parsing results using ParsingEval API.
    """
    model_parsing_score_threse = 0.01

    if metadata.evaluator_type == "sem_seg":
        model_parsing_num_parsing = len(metadata.stuff_classes)
    elif metadata.evaluator_type == "coco":
        classes = metadata.thing_classes
        if "Person" in classes or "Background" in classes:
            model_parsing_num_parsing = len(metadata.thing_classes)
        else:
            model_parsing_num_parsing = len(metadata.thing_classes) + 1
    else:
        raise NotImplementedError(
            "Need to set num parsing !!!"
        )

    pet_eval = ParsingEval(
        parsing_gt,
        semseg_results, parsing_results, part_results, human_results,
        metadata.image_root, output_folder,
        model_parsing_score_threse,
        model_parsing_num_parsing,
        metrics=metrics
    )
    pet_eval.evaluate()
    pet_eval.accumulate()
    pet_eval.summarize()
