import os
import pandas as pd
import numpy as np
import ast

from typing import Tuple, List, Dict
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download

from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern
from . import dbimutils

# 检查推理设备
use_cpu = ('all' in shared.cmd_opts.use_cpu) or ('interrogate' in shared.cmd_opts.use_cpu)

class Interrogator:
    @staticmethod
    def postprocess_tags(
            tags: Dict[str, float],
            threshold=0.35,  # 实时对应 WebUI 滑动条
            additional_tags: List[str] = [],
            exclude_tags: List[str] = [],
            sort_by_alphabetical_order=False,
            add_confident_as_weight=False,
            replace_underscore=False,
            replace_underscore_excludes: List[str] = [],
            escape_tag=False
    ) -> Dict[str, float]:
        # 1. 处理手动添加的标签
        for t in additional_tags:
            tags[t] = 1.0

        # 2. 核心过滤逻辑：仅保留置信度 >= WebUI 阈值的标签
        items = list(tags.items())
        filtered_items = [
            (t, c) for t, c in items
            if c >= threshold and t not in exclude_tags
        ]

        # 3. 排序逻辑
        if sort_by_alphabetical_order:
            # 勾选时：全局 A-Z 排序
            filtered_items.sort(key=lambda i: i[0].lower())
        else:
            # 未勾选：保持 interrogate 传来的 [分类顺序 + 组内置信度降序]
            pass

        # 4. 格式化输出字符串
        res_tags = []
        for tag, confident in filtered_items:
            new_tag = tag
            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')
            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)
            if add_confident_as_weight:
                new_tag = f'({new_tag}:{confident})'
            res_tags.append((new_tag, confident))

        return dict(res_tags)

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tags'):
            del self.tags
        return True

class PixAIInterrogator(Interrogator):
    def __init__(self, name: str, model_path='model.onnx', tags_path='selected_tags.csv', **kwargs) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def load(self) -> None:
        m_path, t_path = self.download()
        from onnxruntime import InferenceSession
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu: providers.pop(0)
        self.model = InferenceSession(str(m_path), providers=providers)

        df = pd.read_csv(t_path)
        df['category'] = df['category'].fillna(0).astype(int)
        df['name'] = df['name'].astype(str)
        df['ips'] = df['ips'].fillna('[]')
        self.tags = df
        print(f'Loaded {self.name}')

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        m_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path))
        t_path = Path(hf_hub_download(**self.kwargs, filename=self.tags_path))
        return m_path, t_path

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        input_details = self.model.get_inputs()[0]
        height = input_details.shape[2]

        # 预处理：高质量缩放以找回 blonde hair 等特征
        image = image.convert('RGB')
        image = image.resize((height, height), resample=Image.LANCZOS)

        image_data = np.asarray(image).astype(np.float32) / 255.0
        image_data = image_data.transpose(2, 0, 1)
        image_data = np.expand_dims(image_data, 0)

        # 推理：使用 Index 1 (Logits) 获取原始高精度数据
        all_outputs = self.model.run(None, {input_details.name: image_data})
        logits = all_outputs[1].flatten()
        # 手动 Sigmoid 转换，确保概率值与官方 Demo 一致
        confidents_raw = 1 / (1 + np.exp(-logits.astype(np.float64)))

        chars_map, ips_map, general_map = {}, {}, {}
        t_names = self.tags['name'].values
        t_cats = self.tags['category'].values
        t_ips = self.tags['ips'].values

        # 遍历并分类提取
        for i in range(len(confidents_raw)):
            prob = float(confidents_raw[i])

            # 此处仅进行极其微小的物理预过滤（0.001），防止性能浪费
            if prob < 0.001: continue

            cat = t_cats[i]
            name = t_names[i]

            if cat == 4: # Character
                chars_map[name] = prob
                if t_ips[i] != '[]':
                    try:
                        for ip in ast.literal_eval(t_ips[i]):
                            ips_map[ip] = max(ips_map.get(ip, 0), prob)
                    except: pass
            elif cat == 0: # General
                general_map[name] = prob

        # 组内置信度降序排序工具（不再限制数量）
        def sort_group(d):
            return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

        # 合并结果：Character -> IP -> General
        combined_results = {}
        combined_results.update(sort_group(chars_map))
        combined_results.update(sort_group(ips_map))
        combined_results.update(sort_group(general_map))

        return {}, combined_results
