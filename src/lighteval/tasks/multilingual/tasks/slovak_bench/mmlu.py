from functools import partial
from string import ascii_uppercase
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.mlmm_mmlu import MMLU_SUBSETS
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.utils.language import Language
from langcodes import standardize_tag
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation, build_choices
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric


LETTER_INDICES = ascii_uppercase


def mmlu_filter(subset, line):
    return line["id"].split("/")[0] == subset


def mmlu_prompt(line, task_name: str = None):
    subject = line["subject"]
    query = f"Nasledujú otázky s výberom odpovedí o {subject.replace('_', ' ')}.\n\nOtázka: {line['question']}"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ascii_uppercase, line["choices"])])
    query += "\nOdpoveď:"

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"Nasledujú otázky s výberom odpovedí o {subject.replace('_', ' ')}.\n\n",
    )

mmlu = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.SLOVAK.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.SLOVAK,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        hf_repo="lighteval/okapi_mmlu",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_filter=partial(mmlu_filter, subset),
        evaluation_splits=("test",),
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=32,
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
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

mmlu_with_prompt = [
    LightevalTaskConfig(
        name=f"mmlu_prompt_{Language.SLOVAK.value}:{subset}",
        prompt_function=mmlu_prompt,
        hf_repo="lighteval/okapi_mmlu",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_filter=partial(mmlu_filter, subset),
        evaluation_splits=("test",),
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=32,
        metrics=[Metrics.exact_match]
    )
    for subset in MMLU_SUBSETS
]

TASKS_TABLE = [
    *mmlu,
    *mmlu_with_prompt,
]
