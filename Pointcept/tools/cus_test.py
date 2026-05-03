from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)

import os
import numpy as np
import torch

from pointcept.datasets import collate_fn
from pointcept.utils.logger import get_root_logger
from pointcept.engines.test import TesterBase
from pointcept.utils.misc import intersection_and_union


class SimpleTester(TesterBase):

    @staticmethod
    def collate_fn(batch):
        return batch

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info("[cus_test] >>>>>>>>>>>>>>>> Start Simple Evaluation (Global Acc / mIoU) >>>>>>>>>>>>>>>>>")

        self.model.eval()

        num_classes = self.cfg.data.num_classes
        ignore_index = self.cfg.data.ignore_index

        # 全局累计器
        intersection_sum = np.zeros(num_classes, dtype=np.float64)
        union_sum = np.zeros(num_classes, dtype=np.float64)
        target_sum = np.zeros(num_classes, dtype=np.float64)

        for idx, data_dict in enumerate(self.test_loader):
            data_dict = data_dict[0]  # 当前假设 batch size == 1
            fragment_list = data_dict.pop("fragment_list")
            segment_full = data_dict.pop("segment")  # voxel 后整帧 GT（index 空间）
            data_name = data_dict.pop("name")
            logger.info(
                f"[cus_test] sample {idx+1}/{len(self.test_loader)}: {data_name}, num_fragments={len(fragment_list)}"
            )

            if len(fragment_list) == 0:
                logger.warning("[cus_test] fragment_list 为空，跳过")
                continue

            # 对齐 SemSegTester：在整帧上累加所有 fragment 的预测
            # pred_full: (N_full, num_classes)
            segment_full_np = (
                segment_full.cpu().numpy()
                if isinstance(segment_full, torch.Tensor)
                else np.asarray(segment_full)
            )
            N_full = segment_full_np.shape[0]
            pred_full = torch.zeros(
                (N_full, num_classes), dtype=torch.float32, device="cuda"
            )

            for fi in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = fi * fragment_batch_size, min(
                    (fi + 1) * fragment_batch_size, len(fragment_list)
                )
                frag_items = fragment_list[s_i:e_i]

                input_dict = collate_fn(frag_items)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                idx_part = input_dict["index"]  # (n_points_frag,)

                with torch.no_grad():
                    output = self.model(input_dict)
                    if not isinstance(output, dict) or "seg_logits" not in output:
                        logger.warning(
                            "[cus_test] model output has no 'seg_logits', skip this fragment."
                        )
                        continue
                    seg_logits = output["seg_logits"]  # (n_points_frag, num_classes)
                    prob = torch.softmax(seg_logits, dim=-1)

                # 将该 fragment 的预测累加回整帧 pred_full
                bs = 0
                for be in input_dict["offset"]:
                    pred_full[idx_part[bs:be], :] += prob[bs:be]
                    bs = be

            # 得到整帧的最终预测标签
            pred_label = pred_full.argmax(dim=-1).cpu().numpy().astype(np.int32)

            # 计算该样本的 intersection / union / target
            intersection, union, target = intersection_and_union(
                pred_label,
                segment_full_np,
                num_classes,
                ignore_index,
            )
            intersection_sum += intersection
            union_sum += union
            target_sum += target

            # 打印当前样本的简单统计
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = float(np.mean(iou_class[mask])) if np.any(mask) else 0.0
            acc = float(np.sum(intersection) / (np.sum(target) + 1e-10))
            logger.info(
                f"[cus_test] sample {idx+1}: Acc={acc:.4f}, mIoU={iou:.4f} (N={N_full})"
            )

        # 计算全局指标
        mask_global = union_sum != 0
        iou_class_global = intersection_sum / (union_sum + 1e-10)
        mIoU_global = float(np.mean(iou_class_global[mask_global])) if np.any(mask_global) else 0.0
        mAcc_global = float(
            np.mean(intersection_sum / (target_sum + 1e-10))
        ) if np.sum(target_sum) > 0 else 0.0
        allAcc_global = float(
            np.sum(intersection_sum) / (np.sum(target_sum) + 1e-10)
        ) if np.sum(target_sum) > 0 else 0.0

        logger.info(
            "[cus_test] Global result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU_global, mAcc_global, allAcc_global
            )
        )

        logger.info(
            "[cus_test] <<<<<<<<<<<<<<<<< End Simple Evaluation <<<<<<<<<<<<<<<<<"
        )


def main():
    """命令行入口：

    用法示例：

        python tools/cus_test.py \
          --config-file configs/so100/xxx.py

    - 解析参数 / 配置；
    - 调用 default_setup 完成环境初始化；
    - 构造 SimpleTester 并运行 test()，按 SemSegTester 的 fragment 推理方式
      在整套测试集上计算并打印整体 mIoU / mAcc / allAcc。
    """
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    default_setup(cfg)

    tester = SimpleTester(cfg, verbose=True)
    tester.test()


if __name__ == "__main__":
    main()

