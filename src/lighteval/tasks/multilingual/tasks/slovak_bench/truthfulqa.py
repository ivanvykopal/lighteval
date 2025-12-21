from functools import partial
from string import ascii_uppercase
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.utils.language import Language
from langcodes import standardize_tag
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation, build_choices
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric


mlmm_truthfulqa_tasks = [ # DONE
    LightevalTaskConfig(
        name=f"truthfulqa_{Language.SLOVAK.value}_{formulation.name.lower()}:{subset}",
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

def truthful_qa_multiple_choice_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Otázka: {line['question']}\nOdpoveď:",
        choices=[f" {c}" for c in line["mc1_targets"]["choices"]] + [f" {c}" for c in line["mc2_targets"]["choices"]],
        gold_index=[
            ix for ix, label in enumerate(line["mc1_targets"]["labels"] + line["mc2_targets"]["labels"]) if label == 1
        ],
        specific={"len_mc1": len(line["mc1_targets"]["choices"])},
    )


def truthful_qa_generative_prompt(line, task_name: str = None):
    correct_answers = [
        answer.strip() + "" if answer[-1] == "." else "." for answer in line["correct_answers"] if answer != ""
    ]
    if "Nemám žiadny komentár." not in correct_answers:
        correct_answers.append("Nemám žiadny komentár.")
    incorrect_answers = [
        answer.strip() + "" if answer[-1] == "." else "." for answer in line["incorrect_answers"] if answer != ""
    ]

    return Doc(
        task_name=task_name,
        query=line["question"].strip(),
        choices=correct_answers + incorrect_answers,
        gold_index=list(range(len(correct_answers))),
        specific={"len_mc1": len(line["mc1_targets"]["choices"])},
    )


truthfulqa_gen = LightevalTaskConfig(
    name=f"truthfulqa_{Language.SLOVAK.value}:gen",
    prompt_function=truthful_qa_generative_prompt,
    hf_repo="jon-tow/okapi_truthfulqa",
    hf_subset=standardize_tag(Language.SLOVAK.value),
    hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
    evaluation_splits=("validation",),
    hf_avail_splits=["validation"],
    few_shots_split=None,
    generation_size=512,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

truthfulqa_mc = LightevalTaskConfig(
    name="truthfulqa:mc",
    prompt_function=truthful_qa_multiple_choice_prompt,
    hf_repo="jon-tow/okapi_truthfulqa",
    hf_subset=standardize_tag(Language.SLOVAK.value),
    hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
    evaluation_splits=("validation",),
    hf_avail_splits=["validation"],
    few_shots_split=None,
    # generation_size=-1,
    metrics=[Metrics.truthfulqa_mc_metrics],
    stop_sequence=["\n"],
    version=0,
)


TASKS_TABLE = [
    *mlmm_truthfulqa_tasks,
    truthfulqa_gen,
    truthfulqa_mc,
]