## Model Details

**Person or organization developing model:** This model was developed by the authors of this paper. Organizations
providing computational support are listed in the Acknowledgements, but this model is not officially developed as part
of any organization. The author affiliations are listed on the first page of this paper.

**Model date:** This model card describes the May 2024 version of TabuLa-8B.

**Model version:** This model card describes version 1.0 of TabuLa-8B.

**Model type:** TabuLa-8B is an autoregressive language model, identical in architecture to Llama 3.

**Information about training algorithms, parameters, fairness constraints or other applied approaches, and features:**
Our training procedure is described in our paper. Our procedure for dataset construction, which includes methods for
removing sensitive PII, is described in our paper.

**Paper or other resource for more information:** This paper is the primary resource for information about TabuLa-8B.
Implementation details can also be found at the open-source code release associated with the project.

**Citation details:** See the first page of this paper.

**License:** The model uses the Meta Llama 3 license (see https://llama.meta.com/llama3/license/).

**Where to send questions or comments about the model:** Send questions or comments directly to the corresponding
authors, or file issues on the project git repo.

## Intended Use

**Primary intended uses:** This is a research-only release. The primary intended use of this model is for research on
tabular data modeling, or for research applications on tabular data.

**Primary intended users:** The primary intended users are scientific researchers interested in understanding, training,
and applying tabular foundation models.

**Out-of-scope use cases:** Commercial use, use of the model to attempt to identify, harm, or violate the privacy of
individuals represented in the training data, and any other behavior that violates the Meta Llama 3 license is out of
scope.

## Factors

**Relevant factors:** The original Model Cards paper identifies factors as ``groups, instrumentation, and environments''
relevant to summaries of model performance. One group relevant to our models' performance is the task type (
classification vs. binned regression). We report performance on these tasks separately; our results are discussed in our
paper. Broadly, we find that TabuLa-8B's overall performance profile relative to baselines is similar for both
classification and binned regression tasks. Similarly, the different benchmarks may be viewed as different
*environments*, each testing a different type of dataset. For example, UniPredict tests performance on datasets with
informative headers; OpenML-CC18 tests performance on datasets without such headers and where traditional supervised
learning methods can be tuned to good performance; Grinsztajn tests performance on datasets where GBDTs tend to perform
best; and AMLB tests performance on tasks including free-form text. Our main results show that TabuLa-8B's overall
performance relative to baselines is similar across these tasks; we analyze the differences in detail in the paper.

**Evaluation factors:** Evaluating language models (LMs) is different from evaluating standard supervised learning
methods: while the latter directly output a score or probability over the set of target labels, LMs only output
next-token probabilities over their vocabularies; as a result, predicted probabilities are not directly available (
although these can be obtained through the use of various heuristics). In order to avoid introducing additional degrees
of freedom into the evaluation process, we do not use score-based evaluation methods that rely on evaluating predicted
probabilities; we only evaluate based on exact matching (as in several works both in the tabular
literature ([LIFT](https://arxiv.org/abs/2206.06565), [TabLLM](https://arxiv.org/abs/2210.10723)) and in the broader
language modeling literature ([PALI](https://arxiv.org/abs/2209.06794), [Flamingo](https://arxiv.org/abs/2204.14198)).
As a consequence, our evaluation does not use metrics which are sometimes used to evaluate tabular classification
models, such as Area Under the Receiver Operating Characteristic Curve (AUC).

## Metrics

**Model performance measures:** Our primary evaluation measures are based on accuracy. We use exact-match accuracy for
language model generations, and top-1 accuracy for supervised learning model predictions.

**Decision thresholds:** We use top-1 accuracy for supervised learning model predictions, but do not apply a specific
threshold.

**Variation approaches:** N/A

## Evaluation Data

**Datasets:** We use a suite of five previously-proposed tabular benchmarks, comprising a total of 329 tables. Our
evaluation datasets are described in detail in our paper.

**Motivation:** Using preexisting benchmark datasets allows us to compare the performance of our models to prior work in
the tabular prediction literature. Additionally, using high-quality, curated benchmarks ensures that we are able to make
reliable conclusion about overall model quality and performance relative to baselines.

**Preprocessing:** Our preprocessing is described in our paper. We perform minimal preprocessing on the datasets (no
one-hot encoding, standardization, etc.) except for the logistic regression baseline, which requires this for best
performance.

## Training Data

Our training data is described in our paper, with further details in the supplementary.

## Quantitative Analyses

**Unitary results:** Our unitary results are summarized in our paper. We provide detailed analysis in the supplementary
section and give per-dataset results in our paper.

**Intersectional results:** We do not explicitly investigate tasks which include sensitive attributes, and so do not
consider intersectional analysis in this work. We call for future work understanding the fairness properties of tabular
foundation models in the future work (and our first-of-its-kind model will enable such research).

## Ethical Considerations

There are important ethical considerations of both the data and model presented in this work. We discuss these in our
Impact Statement.

## Caveats and Recommendations

This model is for research use only. We recommend that more thorough research on both the impact of tabular training
datasets, and the downstream performance of fine-tuned language models, be conducted before the deployment of tabular
foundation models for real-world decisionmaking deployments.