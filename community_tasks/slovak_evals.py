# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

from functools import partial
from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.multilingual.tasks import MMLU_SUBSETS
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function

from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation
from lighteval.utils.language import Language
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from langcodes import standardize_tag

# Hellaswag from ligtheval/mulitlingual
mlmm_hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"community_mlmm_hellaswag_{Language.SLOVAK.value}_{formulation.name.lower()}",
        suite=["community"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.SLOVAK,
            adapter=lambda line: {
                # We don't use activity_label as they are not available
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# MMLU from ligtheval/mulitlingual
mlmm_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"community_mlmm_mmlu_{Language.SLOVAK.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.SLOVAK,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="jon-tow/okapi_mmlu",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="refs/pr/1",
        hf_filter=partial(lambda subset, line: line["id"].split("/")[0] == subset, subset),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# ACR from ligtheval/mulitlingual
mlmm_arc_challenge_tasks = [
    LightevalTaskConfig(
        name=f"community_mlmm_arc_{Language.SLOVAK.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            Language.SLOVAK,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# TruthfulQA from ligtheval/mulitlingual
mlmm_truthfulqa_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_truthfulqa_{Language.SLOVAK.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.SLOVAK,
            partial(
                lambda subset, line: {
                    "question": line["question"],
                    "choices": line[f"{subset}_targets"]["choices"],
                    "gold_idx": [ix for ix, label in enumerate(line[f"{subset}_targets"]["labels"]) if label == 1],  # type: ignore
                },
                subset,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_truthfulqa",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for subset in ["mc1", "mc2"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE = [
    *mlmm_hellaswag_tasks,
    *mlmm_mmlu_tasks,
    *mlmm_arc_challenge_tasks,
    *mlmm_truthfulqa_tasks,
]