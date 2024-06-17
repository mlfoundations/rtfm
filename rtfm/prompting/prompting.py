import io
import json
import os.path
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Sequence, Optional

import pandas as pd
import requests
from tableshift.core import TabularDataset
from tableshift.core.data_source import (
    OfflineDataSource,
    KaggleDataSource,
    KaggleCompetitionDataSource,
    KddCup2009DataSource,
    AutoMLBenchmarkDataSource,
)
from tableshift.core.features import FeatureList, Feature
from ucimlrepo import fetch_ucirepo

from rtfm.data_sources.grinsztajn import (
    SLUG_TO_OPENML_DATASET_ID,
    UCIDataSource,
    DefaultOfCreditCardClientsDataSource,
)
from rtfm.data_sources.uci import fetch_uci_data
from rtfm.prompting.prompt_templates import (
    TASK_SECTION_SEP,
    TASK_SUMMARY_INSTRUCTIONS,
    TASK_CONTEXT_INSTRUCTIONS,
    FEATURE_VARIANTS_INSTRUCTIONS,
    FEATURE_VARIANTS_EXAMPLE,
    TASK_CONTEXT_VARIANTS_INSTRUCTIONS,
    CODE_VARIANTS_INSTRUCTIONS,
    CODE_VARIANTS_EXAMPLE,
)


def get_page_content(url):
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(strip=True)
    except requests.RequestException as e:
        return f"An error occurred: {e}"


from bs4 import BeautifulSoup
import time
from rtfm.data_sources.uci import initialize_webdriver


def get_page_content_with_selenium(url):
    """Use Selenium to fetch the page content.

    This approach (as opposed to purely fetching the content with requests.get()
    is required for most modern webpages that use JavaScript to load their content;
    otherwise requests will simply return a tiny part of the full page."""
    print(f"Getting page content for {url} with selenium")
    # Set up the Selenium WebDriver
    driver = initialize_webdriver()

    try:
        # Fetch the webpage
        driver.get(url)
        # Wait for JavaScript to load
        time.sleep(5)  # Adjust this wait time as necessary

        # Get the page source
        page_source = driver.page_source

        # Parse the HTML content
        soup = BeautifulSoup(page_source, "html.parser")
        return soup.get_text(strip=True)
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        driver.quit()


def extract_urls(s: str) -> Sequence[str]:
    # Updated regex pattern for finding URLs, excluding trailing punctuation
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s<>"]*[^\s<>".,;]'
    # Find all matches of the pattern in the input string
    urls = re.findall(url_pattern, s)
    return urls


def serialize_feature_list(fl: FeatureList) -> str:
    with io.BytesIO() as f:
        fl.to_jsonl(f)
        f.seek(0)
        feature_list_jsonl_text = f.read().decode()
    return feature_list_jsonl_text


@dataclass
class MetadataElement:
    key: str  # short name
    description: str  # long name
    obj: Any  # any object that has a __repr__ method; the value corresponding to this key.


@dataclass
class TaskInfo(ABC):
    tabular_dataset: TabularDataset
    metadata: List[MetadataElement] = field(default_factory=list)
    html_augment: bool = False
    max_characters_per_metadata_elem: Optional[
        int
    ] = 35_000  # appx. 10k tokens for Claude

    @property
    def feature_list(self) -> FeatureList:
        return self.tabular_dataset.task_config.feature_list

    @abstractmethod
    def get_metadata(self) -> Dict[str, str]:
        raise

    def fetch_html_data(self, text):
        """Find any valid links in text, fetch the HTML from them, and append it to metadata."""
        links = extract_urls(text)
        for url in links:
            content = get_page_content_with_selenium(url)
            if content:
                self.metadata.append(
                    MetadataElement(
                        key=url,
                        description=f"HTML content of a page describing the dataset, with URL {url}",
                        obj=content,
                    )
                )

    def serialize_metadata(self) -> str:
        metadata_text = TASK_SECTION_SEP.join(
            f"<{x.key}>\n{x.description}:\n{x.obj}\n</{x.key}>" for x in self.metadata
        )
        # Wrap metadata in <metadata> tags
        metadata_text = (
            "<metadata>"
            + TASK_SECTION_SEP
            + metadata_text
            + TASK_SECTION_SEP
            + "</metadata>"
        )
        return metadata_text

    def make_task_summary_prompt(self, prefix: str = "", suffix: str = "") -> str:
        """Make the task summary prompt by calling self.get_metadata() .

        The task summary is a *structured* set of fields. It takes the (potentially heterogeneous)
        metadata available for a given task, and returns a consistently-formatted set of task information.
        The task summary is mostly used for prompt chaining; for example, it is used in
        generating the task context.
        """
        metadata_text = self.serialize_metadata()

        prompt = TASK_SUMMARY_INSTRUCTIONS.format(
            METADATA=metadata_text, PREFIX=prefix, SUFFIX=suffix
        )
        return prompt

    def make_task_context_prompt(self, prefix: str = "", suffix: str = "") -> str:
        """Make the task context prompt."""
        # check that a task summary is in the metadata
        assert any(
            x.key == "task_summary" for x in self.metadata
        ), "task_summary is required for making the task context prompt."
        metadata_text = self.serialize_metadata()

        prompt = TASK_CONTEXT_INSTRUCTIONS.format(
            METADATA=metadata_text, PREFIX=prefix, SUFFIX=suffix
        )
        return prompt

    def make_feature_variants_prompt(
        self, feature: Feature, prefix: str = "", suffix: str = ""
    ):
        metadata_text = self.serialize_metadata()
        prompt = FEATURE_VARIANTS_INSTRUCTIONS.format(
            METADATA=metadata_text,
            PREFIX=prefix,
            SUFFIX=suffix,
            SERIALIZED_FEATURE=repr(feature),
            FEATURE_VARIANTS_EXAMPLE=FEATURE_VARIANTS_EXAMPLE,
            FEATURE=feature.name_extended if feature.name_extended else feature.name,
        )
        return prompt

    def make_code_variants_prompt(
        self, feature: Feature, prefix: str = "", suffix: str = ""
    ):
        metadata_text = self.serialize_metadata()
        prompt = CODE_VARIANTS_INSTRUCTIONS.format(
            METADATA=metadata_text,
            PREFIX=prefix,
            SUFFIX=suffix,
            FEATURE=feature.name,
            SERIALIZED_FEATURE=repr(feature),
            CODE_VARIANTS_EXAMPLE=CODE_VARIANTS_EXAMPLE,
        )
        return prompt

    def make_context_variants_prompt(self, example, prefix: str = "", suffix: str = ""):
        metadata_text = self.serialize_metadata()
        prompt = TASK_CONTEXT_VARIANTS_INSTRUCTIONS.format(
            METADATA=metadata_text,
            PREFIX=prefix,
            SUFFIX=suffix,
            EXAMPLE=example,
        )
        return prompt


# A string describing the "description" field.
DESCRIPTION_DESCRIPTION = (
    "description of the dataset taken from a website hosting the data"
)
# A string description of the "data" field
DATA_DESCRIPTION = "a sample of the data"
# A string description of the field that contains the results of df.describe().
DATA_DESCRIPTION_DESCRIPTION = "The results of calling pd.Describe() on the dataset."
DATA_TARGET_DESCRIPTION = "Information about the target column in the dataset including overall and relative proportion of each target class"
OPENML_FEATURES_DESCRIPTION = "Information about the features available in the dataset taken from a website hosting the data."
UCI_VARIABLES_DESCRIPTION = "Information about the features available in the dataset taken from a website hosting the data."
UCI_METADATA_DESCRIPTION = "A JSON-formatted string containing metadata about the dataset, taken from a website hosting the data."
FEATURE_LIST_DESCRIPTION = (
    "A Python FeatureList object with detailed information describing the feature, what categories each unique "
    "value maps to, and any additional feature-level metadata. Note that the is_target flag indicates whether "
    "that feature is the prediction target (is_target=True indicates the target)."
)
DATASET_DOCS_DESCRIPTION = (
    "Information about the the dataset taken from a website hosting the data."
)


@dataclass
class OfflineTaskInfo(TaskInfo):
    """Container for offline data source.

    Uses data, description, and the content of dataset documentation."""

    def get_metadata(self) -> Dict[str, str]:
        return {x.description: x.obj for x in self.metadata if x.key != "data"}

    def __post_init__(self):
        df = self.tabular_dataset._get_split_df("train")
        df_description = repr(df.describe(include="all"))
        target_distribution_description = (
            repr(
                df[self.tabular_dataset.task_config.feature_list.target].value_counts()
            )
            + "\n"
            + repr(
                df[self.tabular_dataset.task_config.feature_list.target].value_counts()
                / len(df)
            )
        )
        self.metadata.append(MetadataElement("data", DATA_DESCRIPTION, df))
        self.metadata.append(
            MetadataElement(
                "data_description", DATA_DESCRIPTION_DESCRIPTION, df_description
            )
        )
        self.metadata.append(
            MetadataElement(
                "target_distribution",
                DATA_TARGET_DESCRIPTION,
                target_distribution_description,
            )
        )

        feature_list_jsonl_text = serialize_feature_list(
            self.tabular_dataset.task_config.feature_list
        )
        self.metadata.append(
            MetadataElement(
                "feature_list", FEATURE_LIST_DESCRIPTION, feature_list_jsonl_text
            )
        )
        assert (
            self.tabular_dataset.task_config.feature_list.documentation
        ), "feature list has no documentation!"
        fl_docs = extract_urls(
            self.tabular_dataset.task_config.feature_list.documentation
        )
        if fl_docs:
            for url in fl_docs:
                docs_html = get_page_content_with_selenium(url)
                self.metadata.append(
                    MetadataElement("docs", DATASET_DOCS_DESCRIPTION, docs_html)
                )


@dataclass
class UCITaskInfo(TaskInfo):
    uci_dataset_id: int = None
    metadata_dir: str = "tmp/ucidata/"

    @property
    def dataset_metadata_dir(self):
        """Path to the directory containing task metadata."""
        return os.path.join(self.metadata_dir, str(self.uci_dataset_id))

    def get_metadata(self) -> Dict[str, str]:
        return {x.description: x.obj for x in self.metadata if x.key != "data"}

    def __post_init__(self):
        assert self.uci_dataset_id is not None
        assert os.path.exists(self.metadata_dir), f"{self.metadata_dir} does not exist"
        data = fetch_ucirepo(id=self.uci_dataset_id)
        df = data.data.original
        df_description = repr(df.describe(include="all"))

        self.metadata.append(MetadataElement("data", DATA_DESCRIPTION, df))
        self.metadata.append(
            MetadataElement(
                "data_description", DATA_DESCRIPTION_DESCRIPTION, df_description
            )
        )

        if not os.path.exists(self.dataset_metadata_dir):
            print(
                f"[INFO] directory {self.dataset_metadata_dir} does not exist."
                f"Attempting to fetch UCI data for task {self.uci_dataset_id}."
            )
            fetch_uci_data(dataset_id=self.uci_dataset_id, output_dir=self.metadata_dir)
        else:
            print(f"[INFO] found cached UCI files at {self.dataset_metadata_dir}")

        variables_csv = os.path.join(self.dataset_metadata_dir, "variables.csv")
        if os.path.exists(variables_csv):
            variables_df = pd.read_csv(variables_csv)
            MetadataElement("features", UCI_VARIABLES_DESCRIPTION, repr(variables_df))
        metadata_json = os.path.join(
            self.dataset_metadata_dir, f"metadata_{self.uci_dataset_id}.json"
        )
        if os.path.exists(metadata_json):
            with open(metadata_json, "r") as f:
                metadata_dict = json.load(f)
                self.metadata.append(
                    MetadataElement(
                        "metadata", UCI_METADATA_DESCRIPTION, json.dumps(metadata_dict)
                    )
                )


@dataclass
class OpenMLTaskInfo(TaskInfo):
    openml_dataset_id: int = None
    metadata_dir: str = "./tmp/openml_datasets/"

    @property
    def dataset_metadata_dir(self):
        """Path to the directory containing task metadata."""
        return os.path.join(self.metadata_dir, str(self.openml_dataset_id))

    @property
    def data_file(self):
        """Path to the CSV data file."""
        return os.path.join(self.dataset_metadata_dir, f"{self.openml_dataset_id}.csv")

    def __post_init__(self):
        assert self.openml_dataset_id is not None
        assert os.path.exists(self.metadata_dir), f"{self.metadata_dir} does not exist"
        assert os.path.exists(
            self.dataset_metadata_dir
        ), f"{self.dataset_metadata_dir} does not exist"

        df = pd.read_csv(self.data_file)
        df_description = repr(df.describe(include="all"))

        with open(os.path.join(self.dataset_metadata_dir, "description.txt"), "r") as f:
            description_text = f.read()

        with open(os.path.join(self.dataset_metadata_dir, "features.json"), "r") as f:
            openml_features_text = json.load(f)
        feature_list_jsonl_text = serialize_feature_list(self.feature_list)

        self.metadata.extend(
            [
                MetadataElement(
                    "description", DESCRIPTION_DESCRIPTION, description_text
                ),
                MetadataElement("data", DATA_DESCRIPTION, df),
                MetadataElement(
                    "data_description", DATA_DESCRIPTION_DESCRIPTION, df_description
                ),
                MetadataElement(
                    "features", OPENML_FEATURES_DESCRIPTION, openml_features_text
                ),
                MetadataElement(
                    "feature_list", FEATURE_LIST_DESCRIPTION, feature_list_jsonl_text
                ),
            ]
        )

    def get_metadata(self) -> Dict[str, str]:
        return {x.description: x.obj for x in self.metadata if x.key != "data"}


def make_feature_variants_prompt(
    task_context: str, serialized_feature_list: str
) -> str:
    return (
        f"{FEATURE_VARIANTS_INSTRUCTIONS}"
        + f"{TASK_SECTION_SEP}Before the list of features, here are some more details about the dataset:\n{task_context}"
        f"{TASK_SECTION_SEP}Here is the list of features: {serialized_feature_list}"
    )


def fetch_task_info(
    tabular_dataset, task: str, task_info_kwargs: Optional[Dict[str, Any]] = None
) -> TaskInfo:
    data_source = tabular_dataset.data_source
    if isinstance(data_source, UCIDataSource):
        task_info = UCITaskInfo(tabular_dataset, uci_dataset_id=data_source.id)
    elif isinstance(data_source, OfflineDataSource):
        task_info = OfflineTaskInfo(tabular_dataset)
    elif (
        isinstance(data_source, KaggleDataSource)
        or isinstance(data_source, KaggleCompetitionDataSource)
        or isinstance(data_source, KddCup2009DataSource)
        or isinstance(data_source, AutoMLBenchmarkDataSource)
        or isinstance(data_source, DefaultOfCreditCardClientsDataSource)
    ):
        task_info = OfflineTaskInfo(tabular_dataset)
    elif task in SLUG_TO_OPENML_DATASET_ID:
        if task_info_kwargs is None:
            task_info_kwargs = {
                "metadata_dir": "./tmp/openml_datasets/",
                "openml_dataset_id": SLUG_TO_OPENML_DATASET_ID[task],
            }
        task_info = OpenMLTaskInfo(
            tabular_dataset,
            openml_dataset_id=SLUG_TO_OPENML_DATASET_ID[task],
            **task_info_kwargs,
        )

    else:
        raise ValueError(f"unknown data source {data_source.name}")
    return task_info
