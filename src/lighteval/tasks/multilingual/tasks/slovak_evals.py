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

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric, MultilingualQuasiExactMatchMetric, MultilingualQuasiF1ScoreMetric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.multilingual.tasks import MMLU_SUBSETS
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import MULTI_CHOICE_QA_QUERY, get_mcq_prompt_function

from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.utils.formatting_utils import capitalize, fix_ending_punct
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation, build_choices
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language
from lighteval.tasks.default_prompts import LETTER_INDICES
import lighteval.tasks.default_prompts as prompt
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from langcodes import standardize_tag
from langcodes import Language as LangCodeLanguage
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro, manage_duplicate_language_codes
from lighteval.utils.utils import as_list



# Hellaswag from ligtheval/mulitlingual
mlmm_hellaswag_tasks = [ # DONE
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

def mmlu_filter(subset, line):
    return line["id"].split("/")[0] == subset

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
        hf_repo="lighteval/okapi_mmlu",
        hf_subset=standardize_tag(Language.SLOVAK.value),
        # hf_revision="refs/pr/1",
        # hf_filter=partial(lambda subset, line: line["id"].split("/")[0] == subset, subset),
        hf_filter=partial(mmlu_filter, subset),
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
# mlmm_truthfulqa_tasks = [
#     LightevalTaskConfig(
#         name=f"community_mlmm_truthfulqa_{Language.SLOVAK.value}_{formulation.name.lower()}:{subset}",
#         prompt_function=get_mcq_prompt_function(
#             Language.SLOVAK,
#             partial(
#                 lambda subset, line: {
#                     "question": line["question"],
#                     "choices": line[f"{subset}_targets"]["choices"],
#                     "gold_idx": [ix for ix, label in enumerate(line[f"{subset}_targets"]["labels"]) if label == 1],  # type: ignore
#                 },
#                 subset,
#             ),
#             formulation=formulation,
#         ),
#         suite=("community",),
#         hf_repo="jon-tow/okapi_truthfulqa",
#         hf_subset=standardize_tag(Language.SLOVAK.value),
#         hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
#         evaluation_splits=("validation",),
#         hf_avail_splits=["validation"],
#         metrics=get_metrics_for_formulation(
#             formulation,
#             [
#                 LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
#                 LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
#             ],
#         ),
#     )
#     for subset in ["mc1", "mc2"]
#     for formulation in [
#         MCFFormulation(),
#         CFFormulation(),
#         HybridFormulation(),
#     ]
# ]

# def skanli(line, task_name: str = None):
#     return Doc(
#         task_name=task_name,
#         query=f"{line['premise']}\nOtázka: {line['hypothesis']} Áno, Nie, or Žiaden?\nOdpoveď:",
#         choices=[" Áno", " Žiaden", " Nie"],
#         gold_index=int(line["label"]),
#     )

# sk_anli_lighteval = LightevalTaskConfig(
#     name="community_anli_slk",
#     suite=["community"],
#     prompt_function=prompt.anli,
#     hf_repo="ivykopal/anli_sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "validation", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metrics=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# anli_lighteval_slovak_prompt = LightevalTaskConfig(
#     name="community_anli_sk_prompt",
#     suite=("community",),
#     prompt_function=skanli,
#     hf_repo="ivykopal/anli_sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "validation", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metrics=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# sk_nli_lighteval = LightevalTaskConfig(
#     name="community_nli_slk",
#     suite=["community"],
#     prompt_function=prompt.anli,
#     hf_repo="slovak-nlp/sklep",
#     hf_subset="nli",
#     hf_avail_splits=["train", "validation", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metrics=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# nli_lighteval_slovak_prompt = LightevalTaskConfig(
#     name="community_nli_sk_prompt",
#     suite=("community",),
#     prompt_function=skanli,
#     hf_repo="slovak-nlp/sklep",
#     hf_subset="nli",
#     hf_avail_splits=["train", "validation", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metrics=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# belebele_tasks_slovak = [
#     LightevalTaskConfig(
#         name=f"community_belebele_{language}_{formulation.name.lower()}",
#         prompt_function=get_mcq_prompt_function(
#             iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
#             lambda line: {
#                 "question": line["question"],
#                 "context": line["flores_passage"],
#                 "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
#                 "gold_idx": int(line["correct_answer_num"]) - 1,
#             },
#             formulation=formulation,
#         ),
#         suite=("community",),
#         hf_repo="facebook/belebele",
#         hf_subset=language,
#         evaluation_splits=("test",),
#         hf_avail_splits=["test"],
#         few_shots_select=None,
#         metric=get_metrics_for_formulation(
#             formulation,
#             [
#                 LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
#                 LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
#             ],
#         ),
#     )
#     for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
#     for language in [
#         "slk_Latn"
#     ]
# ]

# sksquad_tasks = [
#     LightevalTaskConfig(
#         name=f"community_skquad_{language.value}",
#         prompt_function=get_qa_prompt_function(
#             language,
#             lambda line: {
#                 "question": line["question"],
#                 "context": line["context"],
#                 "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
#             },
#         ),
#         suite=("community",),
#         hf_repo="TUKE-DeutscheTelekom/skquad",
#         hf_subset="skquad",
#         evaluation_splits=("validation",),
#         hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
#         few_shots_split="train",
#         generation_size=400,
#         stop_sequence=("\n",),
#         metric=(
#             MultilingualQuasiExactMatchMetric(language, "prefix"),
#             MultilingualQuasiF1ScoreMetric(language),
#         ),
#     )
#     for language in [
#         Language.SLOVAK,
#     ]
# ]

# squadsk_tasks = [
#     LightevalTaskConfig(
#         name=f"community_quad_{language.value}",
#         prompt_function=get_qa_prompt_function(
#             language,
#             lambda line: {
#                 "question": line["question"],
#                 "context": line["context"],
#                 "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
#             },
#         ),
#         suite=("community",),
#         hf_repo="TUKE-DeutscheTelekom/squad-sk",
#         hf_subset="plain_text",
#         evaluation_splits=("validation",),
#         hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
#         few_shots_split="train",
#         generation_size=400,
#         stop_sequence=("\n",),
#         metric=(
#             MultilingualQuasiExactMatchMetric(language, "prefix"),
#             MultilingualQuasiF1ScoreMetric(language),
#         ),
#     )
#     for language in [
#         Language.SLOVAK,
#     ]
# ]

# mqa_tasks_slovak = [
#     LightevalTaskConfig(
#         name=f"community_mqa_{language.value}",
#         prompt_function=get_qa_prompt_function(
#             language,
#             lambda line: {
#                 "question": line["name"],
#                 "choices": [line["answers"][0]["text"]],
#             },
#         ),
#         suite=("community",),
#         hf_repo="clips/mqa",
#         hf_subset="sk-all-question",
#         evaluation_splits=("train",),
#         few_shots_select=None,
#         generation_size=400,
#         stop_sequence=("\n",),
#         metric=(
#             MultilingualQuasiExactMatchMetric(language, "prefix"),
#             MultilingualQuasiF1ScoreMetric(language),
#         ),
#     )
#     for language in [
#         Language.SLOVAK,
#     ]
# ]

# webfaq_tasks_slovak = [
#     LightevalTaskConfig(
#         name=f"community_webfaq_{language.value}",
#         prompt_function=get_qa_prompt_function(
#             language,
#             lambda line: {
#                 "question": line["question"],
#                 "choices": [line["answer"]],
#             },
#         ),
#         suite=("community",),
#         hf_repo="PaDaS-Lab/webfaq",
#         hf_subset="slk",
#         evaluation_splits=("default",),
#         few_shots_select=None,
#         generation_size=400,
#         stop_sequence=("\n",),
#         metric=(
#             MultilingualQuasiExactMatchMetric(language, "prefix"),
#             MultilingualQuasiF1ScoreMetric(language),
#         ),
#     )
#     for language in [
#         Language.SLOVAK,
#     ]
# ]

# qa2dsk_tasks = [
#     LightevalTaskConfig(
#         name=f"community_qa2d_{language.value}",
#         prompt_function=get_qa_prompt_function(
#             language,
#             lambda line: {
#                 "question": line["question"],
#                 "choices": [line["answer"], line['turker_answer']],
#             },
#         ),
#         suite=("community",),
#         hf_repo="ctu-aic/qa2d-sk",
#         hf_subset="default",
#         evaluation_splits=("validation",),
#         few_shots_split="train",
#         generation_size=400,
#         stop_sequence=("\n",),
#         metric=(
#             MultilingualQuasiExactMatchMetric(language, "prefix"),
#             MultilingualQuasiF1ScoreMetric(language),
#         ),
#     )
#     for language in [
#         Language.SLOVAK,
#     ]
# ]

# def mathbio(line, task_name: str = None):
#     query = f"The following are multiple choice questions (with answers) about {line['category_en']}.\n\n"
#     query += line["question"] + "\n"
#     query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
#     query += "Answer:"

#     gold_ix = int(line["answer"]) - 1
#     is_few_shots = line.get("__few_shots", False)  # We are adding few shots

#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
#         gold_index=gold_ix,
#         instruction=f"The following are multiple choice questions (with answers) about {line['category_en']}.\n\n",
#     )

# mathbio_slovak = LightevalTaskConfig(
#     name="community_mathbio_sk",
#     suite=("community",),
#     prompt_function=mathbio,
#     hf_repo="dokato/exam-slovak-mathbio",
#     hf_subset="default",
#     hf_avail_splits=["train",],
#     evaluation_splits=["train"],
#     few_shots_select=None,
#     generation_size=1,
#     metric=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# def mathbio_sk(line, task_name: str = None):
#     query = f"Nasledujú otázky s výberom odpovedí (s možnosťou výberu z viacerých odpovedí) týkajúce sa Matematiky.\n\n"
#     query += line["question"] + "\n"
#     query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
#     query += "Odpoveď:"

#     gold_ix = int(line["answer"]) - 1
#     is_few_shots = line.get("__few_shots", False)  # We are adding few shots

#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
#         gold_index=gold_ix,
#         instruction=f"Nasledujú otázky s výberom odpovedí (s možnosťou výberu z viacerých odpovedí) týkajúce sa Matematiky.\n\n",
#     )


# mathbio_slovak_prompt = LightevalTaskConfig(
#     name="community_mathbio_sk_prompt",
#     suite=("community",),
#     prompt_function=mathbio_sk,
#     hf_repo="dokato/exam-slovak-mathbio",
#     hf_subset="default",
#     hf_avail_splits=["train",],
#     evaluation_splits=["train"],
#     few_shots_select=None,
#     generation_size=1,
#     metric=[Metrics.loglikelihood_acc],
#     stop_sequence=["\n"],
#     version=0,
# )

# def boolq_sk(line, task_name: str = None):
#     # remove extra `?`
#     # check if question ends with `?` if not add it
#     question = line["question"].strip()
#     if not question.endswith("?"):
#         question += "?"
#     return Doc(
#         task_name=task_name,
#         query=f"Passage: {line['passage']}\nQuestion: {question}\nAnswer:",
#         choices=[" Yes", " No"],
#         gold_index=[True, False].index(line["answer"]),
#     )
    
# def boolq_sk_prompt(line, task_name: str = None):
#     # remove extra `?`
#     # check if question ends with `?` if not add it
#     question = line["question"].strip()
#     if not question.endswith("?"):
#         question += "?"
#     return Doc(
#         task_name=task_name,
#         query=f"Úryvok: {line['passage']}\nOtázka: {question}\nOdpoveď:",
#         choices=[" Áno", " Nie"],
#         gold_index=[True, False].index(line["answer"]),
#     )

# boolq_slovak = LightevalTaskConfig(
#     name="community_boolq_sk",
#     suite=("community",),
#     prompt_function=boolq_sk,
#     hf_repo="crabz/boolq_sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "validation"],
#     evaluation_splits=["validation"],
#     few_shots_select=None,
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# boolq_slovak_prompt = LightevalTaskConfig(
#     name="community_boolq_sk_prompt",
#     suite=("community",),
#     prompt_function=boolq_sk_prompt,
#     hf_repo="crabz/boolq_sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "validation"],
#     evaluation_splits=["validation"],
#     few_shots_select=None,
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# def senti(line, task_name: str = None):
#     return Doc(
#         task_name=task_name,
#         query=f"Úryvok: {line['text']}\nSentiment: ",
#         choices=["Pozitívne", "Negatívne", "Neutrálne"],
#         gold_index=["positive", "negative", "neutral"].index(line["label"]),
#     )

# senti_sk = LightevalTaskConfig(
#     name="community_senti_sk_prompt",
#     suite=("community",),
#     prompt_function=senti,
#     hf_repo="TUKE-KEMT/senti-sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "test"],
#     evaluation_splits=["test"],
#     few_shots_select=None,
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#         Metrics.f1_score_macro,
#         Metrics.f1_score_micro,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# def slovak_sa(line, task_name: str = None):
#     return Doc(
#         task_name=task_name,
#         query=f"Úryvok: {line['text']}\nSentiment: ",
#         choices=["Negatívne", "Pozitívne"],
#         gold_index=line["label"],
#     )

# slovak_sa_task = LightevalTaskConfig(
#     name="community_slovak_sa_prompt",
#     suite=("community",),
#     prompt_function=slovak_sa,
#     hf_repo="DGurgurov/slovak_sa",
#     hf_subset="default",
#     hf_avail_splits=["train", "validation", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#         Metrics.f1_score_macro,
#         Metrics.f1_score_micro,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# def toxic_sk(line, task_name: str = None):
#     query = f"Je nasledujúci text toxický?\nText: {line['text']}\nOdpoveď:"
    
#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=["Nie", "Áno"],
#         gold_index=line["label"],
#     )

# toxic_sk_task = LightevalTaskConfig(
#     name="community_toxic_sk_prompt",
#     suite=("community",),
#     prompt_function=toxic_sk,
#     hf_repo="TUKE-KEMT/toxic-sk",
#     hf_subset="default",
#     hf_avail_splits=["train", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#         Metrics.f1_score_macro,
#         Metrics.f1_score_micro,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# def hate_speech(line, task_name: str = None):
#     query = f"Predstavuje nasledujúci text hate speech?\nText: {line['text']}\nOdpoveď:"
    
#     return Doc(
#         task_name=task_name,
#         query=query,
#         choices=["Nie", "Áno"],
#         gold_index=line["label"],
#     )

# hatespeech_sk_task = LightevalTaskConfig(
#     name="community_hatespeech_sk_prompt",
#     suite=("community",),
#     prompt_function=hate_speech,
#     hf_repo="TUKE-KEMT/hate_speech_slovak",
#     hf_subset="default",
#     hf_avail_splits=["train", "test"],
#     evaluation_splits=["test"],
#     few_shots_split="train",
#     few_shots_select="random_sampling_from_train",
#     generation_size=5,
#     metric=[
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#         Metrics.f1_score_macro,
#         Metrics.f1_score_micro,
#     ],
#     stop_sequence=["\n"],
#     version=0,
# )

# def get_sib_function(
#     language: Language,
#     formulation: Formulation = MCFFormulation(),
# ):
#     def prompt_fn(line, task_name: str = None):
#         choices = [
#             "geography",
#             "science/technology",
#             "entertainment",
#             "politics",
#             "health",
#             "travel",
#             "sports",
#         ]
        
#         slovak_choices = [
#             "geografia",
#             "veda/technológie",
#             "zábava",
#             "politika",
#             "zdravie",
#             "cestovanie",
#             "šport",
#         ]

#         translation_literals = TRANSLATION_LITERALS[language]

#         instruction_val = "Aká je kategória nasledujúceho textu?"
#         instruction = f"{instruction_val}\n" if instruction_val else ""
        
#         context = ""

#         question = capitalize(fix_ending_punct(line["text"], translation_literals))
#         answers = [capitalize(fix_ending_punct(str(answer), translation_literals)) for answer in slovak_choices]

#         options = build_choices(answers, formulation, translation_literals)
#         options = f"{options}\n" if options else ""
#         answers = build_answers(answers, formulation, translation_literals)

#         answer_word = capitalize(translation_literals.answer)
#         question_word = capitalize(translation_literals.question_word)

#         query = MULTI_CHOICE_QA_QUERY.format(
#             instruction=instruction,
#             question=question,
#             context=context,
#             question_word=question_word,
#             answer_word=answer_word,
#             colon=translation_literals.colon,
#             sentence_space=translation_literals.sentence_space,
#             options=options,
#         )
        
#         # each text has text category as label
#         gold_idx = choices.index(line["category"]) if line["category"] in choices else None

#         return Doc(
#             task_name=task_name,
#             query=query,
#             gold_index=as_list(gold_idx),
#             choices=answers,
#             instruction=instruction_val,
#             unconditioned_query=f"{answer_word}{translation_literals.colon}",
#         )
        
#     return prompt_fn
    
# sib200_sk_task = [
#     LightevalTaskConfig(
#         name=f"community_sib200_sk_prompt_{formulation.name.lower()}",
#         suite=("community",),
#         prompt_function=get_sib_function(
#             Language.SLOVAK,
#             formulation=formulation,
#         ),
#         hf_repo="mteb/sib200",
#         hf_subset="slk_Latn",
#         hf_avail_splits=["train", "test"],
#         evaluation_splits=["test"],
#         few_shots_split="train",
#         few_shots_select="random_sampling_from_train",
#         metric=get_metrics_for_formulation(
#             formulation,
#             [
#                 LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
#                 LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
#                 LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
#             ],
#         ),
#         version=0,
#     )
    
#     for formulation in [
#         MCFFormulation(),
#         CFFormulation(),
#         HybridFormulation(),
#     ]
# ]


TASKS_TABLE = [
    *mlmm_hellaswag_tasks,
    *mlmm_mmlu_tasks,
    *mlmm_arc_challenge_tasks,
    # *mlmm_truthfulqa_tasks,
    # sk_anli_lighteval,
    # anli_lighteval_slovak_prompt,
    # sk_nli_lighteval,
    # nli_lighteval_slovak_prompt,
]