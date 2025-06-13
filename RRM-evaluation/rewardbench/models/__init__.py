# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    MixtralForCausalLM,
    T5ForConditionalGeneration,
)

from .armorm import ArmoRMPipeline
from .beaver import BeaverCostPipeline, BeaverPipeline, LlamaForScore
from .betterpairrm import BetterPairRMPipeline
from .grm import GRewardModel, GRMPipeline
from .inform import INFORMForSequenceClassification
from .internlm import InternLMPipeline
from .ldlreward import LDLPipeline, LDLRewardModel27B
from .openassistant import *  # noqa
from .openbmb import LlamaRewardModel, OpenBMBPipeline
from .pairrm import DebertaV2PairRM, PairRMPipeline
from .pipeline import RewardBenchPipeline
from .qrm import LlamaForRewardModelWithGating3, LlamaForRewardModelWithGating31
from .shp import SHPPipeline
from .slicpairpm import SlicPairPMPipeline
from .starling import (
    LlamaForSequenceClassification,
    StarlingPipeline,
    build_starling_rm,
)
from .ziya import ZiyaPipeline

# Please open a PR if you need to add more custom modeling code / utilize existing code for you model
REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "default_v2": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1": {
        "model_builder": LDLRewardModel27B.from_pretrained,
        "pipeline_builder": LDLPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "torch_dtype": torch.bfloat16,
    },
    "berkeley-nest/Starling-RM-7B-alpha": {
        "model_builder": build_starling_rm,
        "pipeline_builder": StarlingPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Nexusflow/Starling-RM-34B": {
        "model_builder": LlamaForSequenceClassification.from_pretrained,
        "pipeline_builder": StarlingPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "openbmb/UltraRM-13b": {
        "model_builder": LlamaRewardModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "openbmb/Eurus-RM-7b": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "llm-blender/PairRM-hf": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "mightbe/Better-PairRM": {
        "model_builder": DebertaV2PairRM.from_pretrained,
        "pipeline_builder": BetterPairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "stanfordnlp/SteamSHP-flan-t5-xl": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "stanfordnlp/SteamSHP-flan-t5-large": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "IDEA-CCNL/Ziya-LLaMA-7B-Reward": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": ZiyaPipeline,
        "quantized": False,  # handled by .half() in the custom pipeline, as in model card
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v1.0-reward": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v1.0-cost": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverCostPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v2.0-reward": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "PKU-Alignment/beaver-7b-v2.0-cost": {
        "model_builder": LlamaForScore.from_pretrained,
        "pipeline_builder": BeaverCostPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "RLHFlow/pair-preference-model-LLaMA3-8B": {
        "model_builder": AutoModelForCausalLM.from_pretrained,
        "pipeline_builder": SlicPairPMPipeline,
        "quantized": True,
        "custom_dialogue": True,
        "model_type": "Custom Classifier",
    },
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": ArmoRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Sequence Classifier",
        "torch_dtype": torch.bfloat16,
    },
    "Ray2333/GRM-Gemma-2B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Ray2333/GRM-llama3-8B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "internlm/internlm2-20b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "internlm/internlm2-7b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "internlm/internlm2-1_8b-reward": {
        "model_builder": AutoModel.from_pretrained,
        "pipeline_builder": InternLMPipeline,
        "quantized": True,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "nicolinho/QRM-Llama3.1-8B": {
        "model_builder": LlamaForRewardModelWithGating31.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "torch_dtype": torch.bfloat16,
    },
    "nicolinho/QRM-Llama3-8B": {
        "model_builder": LlamaForRewardModelWithGating3.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "torch_dtype": torch.bfloat16,
    },
    "Ray2333/GRM-Gemma2-2B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "Ray2333/GRM-llama3.2-3B-sftreg": {
        "model_builder": GRewardModel.from_pretrained,
        "pipeline_builder": GRMPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
    },
    "infly/INF-ORM-Llama3.1-70B": {
        "model_builder": INFORMForSequenceClassification.from_pretrained,
        "pipeline_builder": RewardBenchPipeline,
        "quantized": False,
        "custom_dialogue": False,
        "model_type": "Seq. Classifier",
        "torch_dtype": torch.bfloat16,
    },
}

DPO_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForCausalLM.from_pretrained,
        "tokenizer_builder": AutoTokenizer.from_pretrained,
    },
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
        "model_builder": MixtralForCausalLM.from_pretrained,
        "tokenizer_builder": LlamaTokenizer.from_pretrained,
    },
}
