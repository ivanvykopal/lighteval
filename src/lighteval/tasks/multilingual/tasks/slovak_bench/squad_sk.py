

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric, MultilingualQuasiF1ScoreMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


squadsk_tasks = [
    LightevalTaskConfig(
        name=f"squad_{Language.SLOVAK.value}",
        prompt_function=get_qa_prompt_function(
            Language.SLOVAK,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        hf_repo="TUKE-DeutscheTelekom/squad-sk",
        hf_subset="plain_text",
        evaluation_splits=("validation",),
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        few_shots_split="train",
        generation_size=512,
        stop_sequence=("\n",),
        metrics=[
            MultilingualQuasiExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.SLOVAK),
        ],
    )
]

TASKS_TABLE = [
    *squadsk_tasks,
]
