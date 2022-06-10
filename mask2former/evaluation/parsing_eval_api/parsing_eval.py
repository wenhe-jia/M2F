import copy, logging
import time

import cv2
import glob
import numpy as np
import os
import warnings
from PIL import Image
from tqdm import tqdm, trange

from .semseg_eval import SemSegEvaluator
from .utils import get_parsing

warnings.filterwarnings("ignore")


class ParsingEval(object):
    """
    Evaluate parsing
    """

    def __init__(
            self,
            parsingGt=None,
            semsegPred=None,
            parsingPred=None,
            partPred=None,
            humanPred=None,
            gt_dir=None,
            pred_dir=None,
            score_thresh=0.001,
            num_parsing=20,
            metrics=('mIoU', 'APp', 'APr')
    ):

        """
        Initialize ParsingEvaluator
        :param parsingGt:   datasets
        :param parsingPred:
        :return: None
        """
        self.parsingGt = parsingGt

        self.semsegPred = semsegPred
        self.parsingPred = parsingPred
        self.partPred = partPred
        self.humanPred = humanPred

        self.params = {}  # evaluation parameters
        self.params = Params(iouType='iou')  # parameters
        self.par_thresholds = self.params.pariouThrs
        self.mask_thresholds = self.params.maskiouThrs
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.score_thresh = score_thresh
        self.num_parsing = num_parsing
        self.metrics = metrics  # ('mIoU , APp , APr')
        self.stats = dict()  # result summarization
        self._logger = logging.getLogger(__name__)

        if 'mIoU' in self.metrics or 'miou' in self.metrics:
            # self.global_parsing_dir = os.path.join(self.pred_dir, 'global_parsing')
            # assert os.path.exists(self.global_parsing_dir)
            self._logger.info('The Global Parsing Images: {}'.format(len(parsingGt)))
            self.semseg_eval = SemSegEvaluator(
                parsingGt, self.semsegPred, self.gt_dir, self.num_parsing,
                gt_dir=self.gt_dir.replace('Images', 'Category_ids')
            )
            self.semseg_eval.evaluate()
            self.semseg_eval.accumulate()
            self.semseg_eval.summarize()
            self.stats.update(self.semseg_eval.stats)
            self._logger.info('=' * 80)

    def _prepare_APp(self):
        self._logger.info('preparing for calucate APp')
        class_recs = dict()
        npos = 0
        image_ids = self.parsingGt.coco.getImgIds()
        image_ids.sort()
        for image_id in image_ids:
            ann_ids = self.parsingGt.coco.getAnnIds(imgIds=image_id, iscrowd=None)
            objs = self.parsingGt.coco.loadAnns(ann_ids)
            # gt_box = []
            parsing_ids = [obj["parsing_id"] for obj in objs if obj['category_id'] == 0]
            anno_adds = get_parsing(
                self.parsingGt.root, self.parsingGt.coco.loadImgs(image_id)[0]['file_name'], parsing_ids
            )
            npos = npos + len(anno_adds)
            det = [False] * len(anno_adds)
            class_recs[image_id] = {'anno_adds': anno_adds, 'det': det}
        self._logger.info('prepare done')
        return class_recs, npos

    def _voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def cal_one_mean_iou(self, gt, pre):
        # gt : single person parsing label map
        # pred : single person parsing pred label map
        k = (gt >= 0) & (gt < self.num_parsing)
        hist = np.bincount(
            self.num_parsing * gt[k].astype(int) + pre[k], minlength=self.num_parsing ** 2
        ).reshape(self.num_parsing, self.num_parsing).astype(np.float)
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / union
        return iu

    def _compute_mask_overlaps(self, pred_masks, gt_masks):
        """
        Computes IoU overlaps between two sets of masks.
        For better performance, pass the largest set first and the smaller second.
        Input:
            pred_masks --  [num_instances, h, width], Instance masks
            gt_masks   --  [num_instances, h, width], ground truth
        """
        pred_areas = self._count_nonzero(pred_masks)
        gt_areas = self._count_nonzero(gt_masks)

        overlaps = np.zeros((pred_masks.shape[0], gt_masks.shape[0]))
        for i in range(overlaps.shape[1]):
            gt_mask = gt_masks[i]
            overlaps[:, i] = self._compute_mask_IoU(gt_mask, pred_masks, gt_areas[i], pred_areas)
        return overlaps

    def _compute_mask_IoU(self, gt_mask, pred_masks, gt_mask_area, pred_mask_areas):
        """
        Calculates IoU of the specific groundtruth mask
        with the array of all the predicted mask.
        Input:
            gt_mask         -- A mask of groundtruth with shape of [h, w].
            pred_masks      -- An array represents a set of masks,
                         with shape [num_instances, h, w].
            gt_mask_area    -- An integer represents the area of gt_mask.
            pred_mask_areas -- A set of integers represents areas of pred_masks.
        """
        # logical_and() can be broadcasting.
        intersection = np.logical_and(gt_mask, pred_masks)
        # True then the corresponding position of output is 1, otherwise is 0.
        intersection = np.where(intersection == True, 1, 0).astype(np.uint8)  # noqa
        intersection = self._count_nonzero(intersection)

        mask_gt_areas = np.full(len(pred_mask_areas), gt_mask_area)
        union = mask_gt_areas + pred_mask_areas[:] - intersection[:]
        iou = intersection / union
        return iou

    def _split_masks(self, instance_img, id_to_convert=None):
        """
        Split a single mixed mask into several class-specified masks.
        Input:
            instance_img  -- An index map with shape [h, w]
              -- A list of instance part IDs that suppose to
                            extract from instance_img, if *None*, extract all the
                            ID maps except for background.
        Return:
            masks -- A collection of masks with shape [num_instance, h, w]
        """
        masks = []

        instance_ids = np.unique(instance_img)
        background_id_index = np.where(instance_ids == 0)[0]
        instance_ids = np.delete(instance_ids, background_id_index)

        if id_to_convert is None:
            for i in instance_ids:
                masks.append((instance_img == i).astype(np.uint8))
        else:
            for i in instance_ids:
                if i in id_to_convert:
                    masks.append((instance_img == i).astype(np.uint8))
        return masks, len(masks)

    def _count_nonzero(self, masks):
        """
        Compute the total number of nonzero items in each mask.
        Input:
            masks -- a three-dimension array with shape [num_instance, h, w],
                    includes *num_instance* of two-dimension mask arrays.
        Return:
            nonzero_count -- A tuple with *num_instance* digital elements,
                            each of which represents the area of specific instance.
        """
        area = []
        for i in masks:
            _, a = np.nonzero(i)
            area.append(a.shape[0])
        area = tuple(area)
        return area

    def _convert2evalformat(self, inst_id_map):
        """
        :param inst_id_map:[h, w]
        :return: masks:[instances,h, w]
        """
        masks = []
        inst_ids = np.unique(inst_id_map)
        # self._logger.info("inst_ids:", inst_ids)
        background_ind = np.where(inst_ids == 0)[0]
        inst_ids = np.delete(inst_ids, background_ind)
        for i in inst_ids:
            im_mask = (inst_id_map == i).astype(np.uint8)  # get single human insseg mask
            masks.append(im_mask)
        return masks, len(inst_ids)

    def _compute_class_apr(self, instance_par_gt_dir, img_name_list, class_id):
        num_IoU_TH = len(self.par_thresholds)
        AP = np.zeros(num_IoU_TH)

        num_gt_masks = 0
        num_pred_masks = 0
        true_pos = []
        false_pos = []
        scores = []

        for i in range(num_IoU_TH):
            true_pos.append([])
            false_pos.append([])

        part_img_cat_dict = {}
        for pred_im in self.partPred:
            if pred_im["img_name"] not in part_img_cat_dict:
                part_img_cat_dict[pred_im["img_name"]] = {}
            if int(pred_im['category_id']) not in part_img_cat_dict[pred_im["img_name"]]:
                part_img_cat_dict[pred_im["img_name"]][int(pred_im['category_id'])] = []
            part_img_cat_dict[pred_im["img_name"]][int(pred_im['category_id'])].append(pred_im)

        for img_name in tqdm(img_name_list, desc='Calculating class: {}..'.format(class_id)):
            instance_img_gt = Image.open(os.path.join(instance_par_gt_dir, img_name + '.png'))
            instance_img_gt = np.array(instance_img_gt)

            # File for accelerating computation.
            # Each line has three numbers: "instance_part_id class_id human_id".
            rfp = open(os.path.join(instance_par_gt_dir, img_name + '.txt'), 'r')
            # Instance ID from groundtruth file.
            gt_part_id = []
            gt_id = []
            for line in rfp.readlines():
                line = line.strip().split(' ')
                gt_part_id.append([int(line[0]), int(line[1])])  # part_id, part_category, discard human_id
                if int(line[1]) == class_id:
                    gt_id.append(int(line[0]))
            rfp.close()

            try:
                partPred_cls = part_img_cat_dict[img_name][class_id]
            except:
                partPred_cls = []

            pred_masks = [x["mask"].toarray().astype(np.uint8) for x in partPred_cls]
            pred_scores = [float(x["score"]) for x in partPred_cls]
            num_pred_instance = len(partPred_cls)

            # Mask for specified class, i.e., *class_id*
            gt_masks, num_gt_instance = self._split_masks(instance_img_gt, set(gt_id))
            num_gt_masks += num_gt_instance
            num_pred_masks += num_pred_instance
            if num_pred_instance == 0:
                continue

            # Collect scores from all the test images that
            # belong to class *class_id*
            scores += pred_scores

            if num_gt_instance == 0:
                for i in range(num_pred_instance):
                    for k in range(num_IoU_TH):
                        false_pos[k].append(1)
                        true_pos[k].append(0)
                continue

            gt_masks = np.stack(gt_masks)
            pred_masks = np.stack(pred_masks)
            # Compute IoU overlaps [pred_masks, gt_makss]
            # overlaps[i, j]: IoU between predicted mask i and gt mask j
            overlaps = self._compute_mask_overlaps(pred_masks, gt_masks)

            max_overlap_index = np.argmax(overlaps, axis=1)
            for i in np.arange(len(max_overlap_index)):
                max_IoU = overlaps[i][max_overlap_index[i]]
                for k in range(num_IoU_TH):
                    if max_IoU > self.par_thresholds[k]:
                        true_pos[k].append(1)
                        false_pos[k].append(0)
                    else:
                        true_pos[k].append(0)
                        false_pos[k].append(1)

        ind = np.argsort(scores)[::-1]
        for k in range(num_IoU_TH):
            m_tp = np.array(true_pos[k])[ind]
            m_fp = np.array(false_pos[k])[ind]

            m_tp = np.cumsum(m_tp)
            m_fp = np.cumsum(m_fp)
            recall = m_tp / float(num_gt_masks)
            precision = m_tp / np.maximum(m_fp + m_tp, np.finfo(np.float64).eps)

            # Compute mean AP over recall range
            AP[k] = self._voc_ap(recall, precision, False)
        return AP

    def computeAPp(self):
        self._logger.info('Evaluating AP^p and PCP')
        class_recs_temp, npos = self._prepare_APp()
        class_recs = [copy.deepcopy(class_recs_temp) for _ in range(len(self.par_thresholds))]

        parsings = []
        scores = []
        image_ids = []
        for idx, p in enumerate(self.parsingPred):
            parsings.append(p['parsing'].toarray())
            scores.append(p['score'])
            image_ids.append(p['image_id'])
        scores = np.array(scores)
        sorted_ind = np.argsort(-scores)

        nd = len(image_ids)
        tp_seg = [np.zeros(nd) for _ in range(len(self.par_thresholds))]
        fp_seg = [np.zeros(nd) for _ in range(len(self.par_thresholds))]
        pcp_list = [[] for _ in range(len(self.par_thresholds))]
        for d in trange(nd, desc='Calculating APp and PCP ..'):
            cur_id = sorted_ind[d]
            if scores[cur_id] < self.score_thresh:
                continue
            R = []
            for j in range(len(self.par_thresholds)):
                R.append(class_recs[j][image_ids[cur_id]])
            ovmax = -np.inf
            jmax = -1

            mask0 = parsings[cur_id]  # curr pred, single person instance
            mask_pred = mask0.astype(np.int)
            mask_gt_u = seg_iou_max = None
            for i in range(len(R[0]['anno_adds'])):
                mask_gt = R[0]['anno_adds'][i]
                seg_iou = self.cal_one_mean_iou(mask_gt, mask_pred.astype(np.uint8))

                mean_seg_iou = np.nanmean(seg_iou)
                if mean_seg_iou > ovmax:
                    ovmax = mean_seg_iou
                    seg_iou_max = seg_iou
                    jmax = i
                    mask_gt_u = np.unique(mask_gt)

            for j in range(len(self.par_thresholds)):
                if ovmax > self.par_thresholds[j]:
                    if not R[j]['det'][jmax]:
                        tp_seg[j][d] = 1.
                        R[j]['det'][jmax] = 1
                        pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u > 0, mask_gt_u < self.num_parsing)])
                        pcp_n = float(np.sum(seg_iou_max[1:] > self.par_thresholds[j]))
                        if pcp_d > 0:
                            pcp_list[j].append(pcp_n / pcp_d)
                        else:
                            pcp_list[j].append(0.0)
                    else:
                        fp_seg[j][d] = 1.
                else:
                    fp_seg[j][d] = 1.

        # compute precision recall
        all_APp = {}
        all_PCP = {}
        for j, thre in enumerate(self.par_thresholds):
            fp_seg[j] = np.cumsum(fp_seg[j])
            tp_seg[j] = np.cumsum(tp_seg[j])
            rec_seg = tp_seg[j] / float(npos)
            prec_seg = tp_seg[j] / np.maximum(tp_seg[j] + fp_seg[j], np.finfo(np.float64).eps)
            APp = self._voc_ap(rec_seg, prec_seg)
            all_APp[thre] = APp

            assert (np.max(tp_seg[j]) == len(pcp_list[j])), "%d vs %d" % (np.max(tp_seg[j]), len(pcp_list[j]))
            pcp_list[j].extend([0.0] * (npos - len(pcp_list[j])))
            PCP = np.mean(pcp_list[j])
            all_PCP[thre] = PCP
        return all_APp, all_PCP

    def computeAPr(self):
        self._logger.info('Evaluating AP^r')
        instance_par_gt_dir = self.gt_dir.replace('Images', 'Instance_ids')
        assert os.path.exists(instance_par_gt_dir)

        tmp_instance_par_gt_dir = instance_par_gt_dir
        img_name_list = [x.split("/")[-1].split(".")[0] for x in
                         glob.glob(tmp_instance_par_gt_dir + '/*') if x[-3:] == 'txt']

        APr = np.zeros((self.num_parsing - 1, len(self.par_thresholds)))
        with tqdm(total=self.num_parsing - 1) as pbar:
            pbar.set_description('Calculating AP^r ..')
            for class_id in range(1, self.num_parsing):
                APr[class_id - 1, :] = self._compute_class_apr(
                    instance_par_gt_dir, img_name_list, class_id
                )
                pbar.update(1)

        # AP under each threshold.
        mAPr = np.nanmean(APr, axis=0)
        APr_cat = np.nanmean(APr, axis=1)
        all_APr = {}
        for i, thre in enumerate(self.par_thresholds):
            all_APr[thre] = mAPr[i]
        return all_APr, APr_cat

    def computeAPh(self):
        self._logger.info('Evaluating AP^h')
        instance_seg_gt_dir = self.gt_dir.replace('Images', 'Human_ids')
        assert os.path.exists(instance_seg_gt_dir)

        iou_thre_num = len(self.mask_thresholds)

        gt_mask_num = 0
        pre_mask_num = 0
        tp = []
        fp = []
        scores = []
        for i in range(iou_thre_num):
            tp.append([])
            fp.append([])

        tmp_instance_seg_gt_dir = instance_seg_gt_dir
        img_name_list = [x.split("/")[-1].split(".")[0] for x in glob.glob(tmp_instance_seg_gt_dir + '/*')]

        for img_name in tqdm(img_name_list, desc='Calculating APh..'):
            gt_mask = cv2.imread(os.path.join(instance_seg_gt_dir, img_name + '.png'), 0)
            gt_mask, n_gt_inst = self._convert2evalformat(gt_mask)

            humanPred_im = [x for x in self.humanPred if x["img_name"] == img_name]
            pre_mask = [x["mask"].toarray().astype(np.uint8) for x in humanPred_im]
            tmp_scores = [float(x["score"]) for x in humanPred_im]
            n_pre_inst = len(humanPred_im)

            gt_mask_num += n_gt_inst
            pre_mask_num += n_pre_inst

            if n_pre_inst == 0:
                continue

            scores += tmp_scores  # person instance scores for curr image

            if n_gt_inst == 0:
                for i in range(n_pre_inst):
                    for k in range(iou_thre_num):
                        fp[k].append(1)
                        tp[k].append(0)
                continue

            gt_mask = np.stack(gt_mask)
            pre_mask = np.stack(pre_mask)
            overlaps = self._compute_mask_overlaps(pre_mask, gt_mask)  # (num_pred, num_gt)
            max_overlap_ind = np.argmax(overlaps, axis=1)
            for i in np.arange(len(max_overlap_ind)):
                max_iou = overlaps[i][max_overlap_ind[i]]
                for k in range(iou_thre_num):
                    if max_iou > self.mask_thresholds[k]:
                        tp[k].append(1)
                        fp[k].append(0)
                    else:
                        tp[k].append(0)
                        fp[k].append(1)

        all_APh = {}
        ind = np.argsort(scores)[::-1]
        for k in range(iou_thre_num):
            m_tp = tp[k]
            m_fp = fp[k]
            m_tp = np.array(m_tp)
            m_fp = np.array(m_fp)
            m_tp = m_tp[ind]
            m_fp = m_fp[ind]
            m_tp = np.cumsum(m_tp)
            m_fp = np.cumsum(m_fp)
            recall = m_tp / float(gt_mask_num)
            precition = m_tp / np.maximum(m_fp + m_tp, np.finfo(np.float64).eps)
            all_APh[self.mask_thresholds[k]] = self._voc_ap(recall, precition, False)
        return all_APh

    def evaluate(self):
        self._logger.info('Evaluating Parsing predictions')
        if 'APp' in self.metrics or 'ap^p' in self.metrics:
            APp, PCP = self.computeAPp()
            self.stats.update(dict(APp=APp, PCP=PCP))
        if 'APr' in self.metrics or 'ap^r' in self.metrics:
            APr, APr_cat = self.computeAPr()
            self.stats.update(dict(APr=APr))
            self.stats.update(dict(APr_cat=APr_cat))
        if 'APh' in self.metrics or 'ap^h' in self.metrics:
            APh = self.computeAPh()
            self.stats.update(dict(APh=APh))

    def accumulate(self, p=None):
        pass

    def summarize(self):
        if 'APp' in self.metrics or 'ap^p' in self.metrics:
            APp = self.stats['APp']
            PCP = self.stats['PCP']
            mAPp = np.nanmean(np.array(list(APp.values())))
            self._logger.info('~~~~ Summary metrics ~~~~')
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'.format(mAPp))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'.format(APp[0.1]))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'.format(APp[0.3]))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'.format(APp[0.5]))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.60      ] = {:.3f}'.format(APp[0.6]))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'.format(APp[0.7]))
            self._logger.info(
                ' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'.format(APp[0.9]))
            self._logger.info(
                ' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'.format(PCP[0.5]))
            self._logger.info('=' * 80)

        if 'APr' in self.metrics or 'ap^r' in self.metrics:
            APr = self.stats['APr']
            mAPr = np.nanmean(np.array(list(APr.values())))
            APr_cat = self.stats['APr_cat']
            self._logger.info('~~~~ Summary metrics (per category)~~~~')
            for cat_id, apr_c in enumerate(APr_cat):
                self._logger.info(
                    ' Average Precision based on region (APr)' +'Class '+
                    str(cat_id + 1) + '         @[mIoU=0.10:0.90 ] = {:.3f}'.format(apr_c)
                )
            self._logger.info('='*80)
            self._logger.info('~~~~ Summary metrics ~~~~')
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.10:0.90 ] = {:.3f}'.format(mAPr))
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.10      ] = {:.3f}'.format(APr[0.1]))
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.30      ] = {:.3f}'.format(APr[0.3]))
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.50      ] = {:.3f}'.format(APr[0.5]))
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.70      ] = {:.3f}'.format(APr[0.7]))
            self._logger.info(
                ' Average Precision based on region (APr)             @[mIoU=0.90      ] = {:.3f}'.format(APr[0.9]))
            self._logger.info('=' * 80)

        if 'APh' in self.metrics or 'ap^h' in self.metrics:
            APh = self.stats['APh']
            mAPh = np.nanmean(np.array(list(APh.values())))
            self._logger.info('~~~~ Summary metrics ~~~~')
            self._logger.info(
                ' Average Precision based on human (APh)             @[mIoU=0.50:0.95 ] = {:.3f}'.format(mAPh))
            self._logger.info(
                ' Average Precision based on human (APh)             @[mIoU=0.50      ] = {:.3f}'.format(APh[0.5]))
            self._logger.info(
                ' Average Precision based on human (APh)             @[mIoU=0.75      ] = {:.3f}'.format(APh[0.75]))
            self._logger.info('=' * 80)


class Params:
    """
    Params for coco evaluation api
    """

    def setParsingParams(self):
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.pariouThrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.maskiouThrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.maxDets = [None]
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all']
        self.useCats = 1

    def __init__(self, iouType='iou'):
        if iouType == 'iou':
            self.setParsingParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType


def generate_parsing_result(parsings, instance_scores, part_scores, bbox_scores=None, semseg=None, img_info=None,
                            output_folder=None, score_thresh=0.05, semseg_thresh=0.3, parsing_nms_thres=1.0,
                            num_parsing=20):
    parsings = np.asarray(parsings)
    instance_scores = np.asarray(instance_scores)
    part_scores = np.asarray(part_scores)
    bbox_scores = np.asarray(bbox_scores) if bbox_scores is not None else instance_scores
    global_parsing_dir = os.path.join(output_folder, 'global_parsing')
    if not os.path.exists(global_parsing_dir):
        os.makedirs(global_parsing_dir)
    ins_semseg_dir = os.path.join(output_folder, 'instance_segmentation')
    if not os.path.exists(ins_semseg_dir):
        os.makedirs(ins_semseg_dir)
    ins_parsing_dir = os.path.join(output_folder, 'instance_parsing')
    if not os.path.exists(ins_parsing_dir):
        os.makedirs(ins_parsing_dir)

    im_name = img_info['file_name']
    if '/' in im_name:
        folders = im_name.split('/')[:-1]
        cur_global_parsing_dir = global_parsing_dir
        cur_ins_semseg_dir = ins_semseg_dir
        cur_ins_parsing_dir = ins_parsing_dir
        for f_name in folders:
            os.mkdir(os.path.join(cur_global_parsing_dir, f_name)) \
                if not os.path.exists(os.path.join(cur_global_parsing_dir, f_name)) else None
            os.mkdir(os.path.join(cur_ins_semseg_dir, f_name)) \
                if not os.path.exists(os.path.join(cur_ins_semseg_dir, f_name)) else None
            os.mkdir(os.path.join(cur_ins_parsing_dir, f_name)) \
                if not os.path.exists(os.path.join(cur_ins_parsing_dir, f_name)) else None
            cur_global_parsing_dir = cur_global_parsing_dir + '/' + f_name
            cur_ins_semseg_dir = cur_ins_semseg_dir + '/' + f_name
            cur_ins_parsing_dir = cur_ins_parsing_dir + '/' + f_name
    save_global_parsing = os.path.join(global_parsing_dir, im_name.replace('jpg', 'png'))
    save_ins_semseg = os.path.join(ins_semseg_dir, im_name.replace('jpg', 'png'))
    save_ins_parsing = os.path.join(ins_parsing_dir, im_name.replace('jpg', 'png'))

    if semseg is not None:
        semseg = cv2.resize(semseg, (img_info["width"], img_info["height"]), interpolation=cv2.INTER_LINEAR)
        parsing_max = np.max(semseg, axis=2)
        max_map = np.where(parsing_max > 0.7, 1, 0)
        global_parsing = np.argmax(semseg, axis=2).astype(np.uint8) * max_map
    else:
        global_parsing = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
    global_for_ins = np.copy(global_parsing)
    ins_semseg = np.zeros_like(global_parsing, dtype=np.uint8)
    ins_parsing = np.zeros_like(global_parsing, dtype=np.uint8)
    is_wfp = open(save_ins_semseg.replace('png', 'txt'), 'w')
    ip_wfp = open(save_ins_parsing.replace('png', 'txt'), 'w')

    # generate global_parsing, which is semseg
    sorted_bbox_scores_ids = bbox_scores.argsort()
    for s_id in sorted_bbox_scores_ids:
        if bbox_scores[s_id] < semseg_thresh:
            continue
        cur_parsing = parsings[s_id]  # single person parsing label map
        global_parsing = np.where(cur_parsing > 0, cur_parsing,
                                  global_parsing)  # get part semseg label map for single image

    # parsing nms
    if parsing_nms_thres < 1.0:
        parsings, instance_scores, part_scores = parsing_nms(
            parsings, instance_scores, part_scores, parsing_nms_thres, num_parsing
        )

    # generate ins_semseg and global_for_ins
    sorted_score_ids = instance_scores.argsort()
    ins_id = 1
    filtered_part_scores = dict()
    det_bboxes = []
    for s_id in sorted_score_ids:
        if instance_scores[s_id] < score_thresh:
            continue
        cur_parsing = parsings[s_id]  # single person parsing label map
        global_for_ins = np.where(cur_parsing > 0, cur_parsing,
                                  global_for_ins)  # get semseg label map for single image at image scene
        ins_semseg = np.where(cur_parsing > 0, ins_id, ins_semseg)  # get person instance map
        cur_bbox = cv2.boundingRect(cur_parsing.copy())
        x, y, w, h = cur_bbox
        filtered_part_scores[ins_id] = [p for p in part_scores[s_id]]
        det_bboxes.append([instance_scores[s_id], y, x, y + h, x + w])  # for VIP format
        ins_id += 1

    # generate ins_parsing
    ins_ids = np.unique(ins_semseg)
    bg_id_index = np.where(ins_ids == 0)[0]
    ins_ids = np.delete(ins_ids, bg_id_index)
    total_part_num = 0
    for idx in ins_ids:
        part_label = (np.where(ins_semseg == idx, 1, 0) * global_for_ins).astype(
            np.uint8)  # get single person parsing map at image scene
        part_classes = np.unique(part_label)
        for part_id in part_classes:
            if part_id == 0:
                continue
            total_part_num += 1
            if total_part_num >= 255:
                ins_parsing[np.where(part_label == part_id)] = 0
            else:
                ins_parsing[np.where(part_label == part_id)] = total_part_num
                ip_wfp.write('{} {}\n'.format(part_id, filtered_part_scores[idx][part_id - 1]))

    reindex_ins_semseg = np.zeros_like(ins_semseg, dtype=np.uint8)
    re_ins_id = 1
    for idx in ins_ids:
        reindex_ins_semseg = np.where(ins_semseg == idx, re_ins_id, reindex_ins_semseg)
        is_wfp.write('{} {} {} {} {}\n'.format(
            det_bboxes[idx - 1][0], det_bboxes[idx - 1][1], det_bboxes[idx - 1][2], det_bboxes[idx - 1][3],
            det_bboxes[idx - 1][4])
        )
        re_ins_id += 1

    cv2.imwrite(save_global_parsing, global_parsing.astype(np.uint8))  # mIoU
    cv2.imwrite(save_ins_semseg, reindex_ins_semseg.astype(np.uint8))  # APh
    cv2.imwrite(save_ins_parsing, ins_parsing.astype(np.uint8))  # APr
    is_wfp.close()
    ip_wfp.close()

    return parsings, instance_scores


def parsing_nms(parsings, instance_scores, part_scores=None, nms_thresh=0.6, num_parsing=20):
    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def cal_one_mean_iou(image_array, label_array, _num_parsing):
        hist = fast_hist(label_array, image_array, _num_parsing).astype(np.float)
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / union
        return iu

    def parsing_iou(src, dsts, num_classes=20):
        ious = []
        for d in range(dsts.shape[0]):
            iou = cal_one_mean_iou(src, dsts[d], num_classes)
            ious.append(np.nanmean(iou))
        return ious

    sorted_ids = (-instance_scores).argsort()
    sorted_parsings = parsings[sorted_ids]
    sorted_instance_scores = instance_scores[sorted_ids]
    if part_scores is not None:
        sorted_part_scores = part_scores[sorted_ids]
    else:
        raise NotImplementedError
    keepped = [True] * sorted_instance_scores.shape[0]
    for i in range(sorted_instance_scores.shape[0] - 1):
        if not keepped[i]:
            continue
        ious = parsing_iou(sorted_parsings[i], (sorted_parsings[i + 1:])[keepped[i + 1:]], num_parsing)
        for idx, iou in enumerate(ious):
            if iou >= nms_thresh:
                keepped[i + 1 + idx] = False
    parsings = sorted_parsings[keepped]
    instance_scores = sorted_instance_scores[keepped]
    if part_scores is not None:
        part_scores = sorted_part_scores[keepped]

    return parsings, instance_scores, part_scores
