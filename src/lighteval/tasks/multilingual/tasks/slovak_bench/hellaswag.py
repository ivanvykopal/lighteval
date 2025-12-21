from string import ascii_uppercase
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from langcodes import standardize_tag
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation, build_choices
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric


def hellaswag_slovak_prompt_optimized(line, task_name: str = None):
    INSTRUCTION = "Nasledujú otázky s výberom odpovedí.\n\n"
    
    endings = line["endings"]
    num_endings = len(endings)
    
    query = (
        f"{INSTRUCTION}"
        f"Otázka: {line['activity_label']}: "
        f"{line['ctx_a']} {line['ctx_b'].capitalize()}\n"
        + "".join(
            f"{ascii_uppercase[i]}. {endings[i]}\n"
            for i in range(num_endings)
        )
        + "Odpoveď:"
    )
    
    label = line["label"]
    gold_ix = int(label) if label != "" else -1

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {ascii_uppercase[i]}" for i in range(num_endings)],
        gold_index=gold_ix,
        instruction=INSTRUCTION,
    )


hellaswag = [ # DONE
    LightevalTaskConfig(
        name=f"community_mlmm_hellaswag_{Language.SLOVAK.value}_{formulation.name.lower()}",
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


TASKS_TABLE = [
    *hellaswag,
    LightevalTaskConfig(
        name=f"community_mlmm_hellaswag_{Language.SLOVAK.value}",
        prompt_function=hellaswag_slovak_prompt_optimized,
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        metrics=[
            Metrics.exact_match,
        ],
    )
]
