from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from string import ascii_uppercase
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.utils.language import Language
from langcodes import standardize_tag
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation, build_choices
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric


LETTER_INDICES = ascii_uppercase

def arc_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Otázka: {line['question']}\nOdpoveď:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def record_to_sample(record):
    query = record["question"].strip()
    target = record["answerKey"]
    choices = record["choices"]["text"]

    return Sample(input=query, target=target, choices=choices)


mlmm_arc_challenge_tasks = [ # DONE
    LightevalTaskConfig(
        name=f"arc_{Language.SLOVAK.value}_{formulation.name.lower()}:challenge",
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
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=32,
        sample_fields=record_to_sample,
        solver=[multiple_choice(cache=True)],
        scorer=choice(),
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

arc_challenge = LightevalTaskConfig(
    name=f"arc_{Language.SLOVAK.value}:challenge",
    prompt_function=arc_prompt,
    hf_repo="jon-tow/okapi_arc_challenge",
    hf_subset=standardize_tag(Language.SLOVAK.value),
    hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
    evaluation_splits=("test",),
    few_shots_split="train",
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=32,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    arc_challenge,
    *mlmm_arc_challenge_tasks,
]