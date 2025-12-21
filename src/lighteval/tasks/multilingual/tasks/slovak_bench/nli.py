from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from string import ascii_uppercase
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.anli import anli_prompt, record_to_sample


def record_to_sample_sk(record):
    choices = ["Áno", "Žiaden", "Nie"]
    query = f"{record['premise']}\nOtázka: {record['hypothesis']}"
    return Sample(input=query, target=ascii_uppercase[record["label"]], choices=choices)


def skanli(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['premise']}\nOtázka: {line['hypothesis']} Áno, Nie, or Žiaden?\nOdpoveď:",
        choices=[" Áno", " Žiaden", " Nie"],
        gold_index=int(line["label"]),
    )


sk_anli_lighteval = LightevalTaskConfig(
    name="anli_slk",
    prompt_function=anli_prompt,
    hf_repo="ivykopal/anli_sk",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32,
    sample_fields=record_to_sample,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)


anli_lighteval_slovak_prompt = LightevalTaskConfig(
    name="anli_slk_prompt",
    prompt_function=skanli,
    hf_repo="ivykopal/anli_sk",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

sk_nli_lighteval = LightevalTaskConfig(
    name="nli_slk",
    prompt_function=anli_prompt,
    hf_repo="slovak-nlp/sklep",
    hf_subset="nli",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

nli_lighteval_slovak_prompt = LightevalTaskConfig(
    name="nli_slk_prompt",
    prompt_function=skanli,
    hf_repo="slovak-nlp/sklep",
    hf_subset="nli",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=32,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

TASKS_TABLE = [
    sk_anli_lighteval,
    anli_lighteval_slovak_prompt,
    sk_nli_lighteval,
    nli_lighteval_slovak_prompt,
]
