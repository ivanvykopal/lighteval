from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric, MultilingualQuasiF1ScoreMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


webfaq_tasks_slovak = [
    LightevalTaskConfig(
        name=f"webfaq_{Language.SLOVAK.value}",
        prompt_function=get_qa_prompt_function(
            Language.SLOVAK,
            lambda line: {
                "question": line["question"],
                "choices": [line["answer"]],
            },
        ),
        hf_repo="PaDaS-Lab/webfaq",
        hf_subset="slk",
        evaluation_splits=("default",),
        few_shots_select=None,
        generation_size=512,
        stop_sequence=("\n",),
        metrics=[
            MultilingualQuasiExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.SLOVAK),
        ],
    )
]

TASKS_TABLE = [
    *webfaq_tasks_slovak,
]
