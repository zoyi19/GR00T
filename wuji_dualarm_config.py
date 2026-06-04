# wuji_dualarm_config.py
# GR00T N1.7 自定义本体 modality 配置:双臂(7+7=14)+ 舞肌灵巧手(20)= 34 维关节空间
# 用法:gr00t/experiment/launch_finetune.py --embodiment-tag NEW_EMBODIMENT \
#       --modality-config-path /home/neotix/noetix/wuji_example/wuji_dualarm_config.py
#
# key 名严格对应 processed_data/meta/modality.json:
#   state:  right_arm[0:7]  left_arm[7:14]  hand_actual[14:34]
#   action: right_arm[0:7]  left_arm[7:14]  hand_target[14:34]
#   video:  head, wrist     annotation: human.action.task_description
# 与本机 examples/SO100/so100_config.py 同结构(关节臂 RELATIVE / NON_EEF)。

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

wuji_dualarm_config = {
    # 相机:只用当前帧;key 必须与 modality.json 的 "video" 段一致
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "wrist"],
    ),
    # 本体状态:当前帧;key 必须与 modality.json 的 "state" 段一致
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_arm", "left_arm", "hand_actual"],
        # 关节角是弧度,默认 min/max 归一化即可。若想给臂关节做 sin/cos 编码,
        # 取消下一行注释(注意:sin/cos 不可逆,绝不能用于 action):
        # sin_cos_embedding_keys=["right_arm", "left_arm"],
    ),
    # 动作:预测未来 16 步;每个 modality key 对应一个 ActionConfig(顺序一致)
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # horizon = 16,与 relative_stats.json 的 [16, N] 一致
        modality_keys=["right_arm", "left_arm", "hand_target"],
        action_configs=[
            # 右臂:RELATIVE = 相对当前状态的增量(关节空间;泛化更好,推理时自动解回绝对目标)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 左臂:同上
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 灵巧手 20 维:RELATIVE;因 action key(hand_target)与 state key(hand_actual)不同名,
            # 必须显式指定参考状态 state_key="hand_actual",否则 RELATIVE 找不到参考帧会报错。
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="hand_actual",
            ),
        ],
    ),
    # 语言:任务指令(带 annotation. 前缀,文本本身在 meta/tasks.jsonl)
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(wuji_dualarm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# ── 备选:若想用 ABSOLUTE(绝对关节目标,部署最直接、无漂移)──
# 把上面三个 ActionConfig 的 rep 全改成 ActionRepresentation.ABSOLUTE,并删掉 hand 的 state_key。
# 此时 relative_stats.json 会变空 {},归一化全走 stats.json 的 per-dim min/max;
# 改动后必须重跑 gr00t/data/stats.py 重新生成统计量。
