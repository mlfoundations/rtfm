TASK_SECTION_SEP = "\n\n##############\n\n"
TASK_SUMMARY_INSTRUCTIONS = f"""{{PREFIX}}
You are being asked to summarize a variety of metadata about a dataset for users who will utilize that data to perform a prediction task.
Below, after the task instructions, I will provide this metadata in a series of distinct sections. 
Each section starts with the characters '{TASK_SECTION_SEP.strip()}' and a short textual description of the metadata, before providing the metadata itself.

You are a helpful but concise, matter-of-fact, and scientific AI assistant, and should maintain a scientific tone.
Read the metadata below carefully, because I'm going to ask you to to produce a JSON-structured output summarizing the metadata.

{{METADATA}}

Instructions: Your task is to produce a JSON-structured output designed to provide an overview of the dataset for scientific study of the prediction task described. 
The JSON file should contain the following keys, along with a few sentences of descriptive text, following the schema below:
<output-schema>
- Source: <Describe the source of the data.>
- Collection: <Describe how the data was collected and what a single unit/row of the data represents.>
- Population: <Describe the population represented in the dataset. This should clearly describe who or what population the sample was taken from, and therefore what population is represented in the data. If necessary, also describe who or what was excluded from the dataset.>
- Time range: <Describe the time range over which the data was collected.>
- Features: <Give a description of the features in the dataset. This can include descriptions of individual features, and can also describe any conceptual groupings of related features.>
- Prediction target: <Describe the prediction targets/class labels here. Give an extensive description, if possible.>
- Class distribution: <Describe the distribution of the classes in the dataset. Make sure to clearly identify the majority class, if possible.>
- Predictive Variables: <Describe which variables are known to be predictive of the outcome, based on the metadata or your own scientific knowledge of this task or similar tasks.>
</output-schema>

Rules:
<rules>
- Be precise and detailed.
- Skip the preamble, go straight to the JSON output.
- Return only a JSON-structured output, where the keys are strings and the values are strings. NEVER return any other text besides the JSON.
- Using the provided information, give as much detail as possible in each value.
- Only use the provided information. If there is no sufficient information provided, state that this information is not provided.
- Provide all details necessary for a scientist that might use the dataset for their own research or experiments.
- Describe all context needed for someone to understand the dataset.
- Give a detailed description of any factors which might be related to the prediction target (either those present in the dataset, or not present in the dataset).
- Give a precise technical description of any context surrounding this dataset which might be relevant to understanding it; refer to specific information when providing context (i.e. specific events or facts related to the prediction target, population in the dataset, etc.), not general descriptions.
</rules>

Think about your answer first before you respond.

IMPORTANT: return only JSON as output.

{{SUFFIX}}
"""

TASK_CONTEXT_INSTRUCTIONS = f"""{{PREFIX}}
You are being asked to summarize a variety of metadata about a dataset for users who will utilize that data to perform a prediction task.
Below, after the task instructions, I will provide this metadata in a series of distinct sections. 
Each section starts with the characters '{TASK_SECTION_SEP.strip()}' and a short textual description of the metadata, before providing the metadata itself.

You are a helpful but concise, matter-of-fact, and scientific AI assistant, and should maintain a scientific tone.
Read the metadata below carefully, because I'm going to ask you to to produce output summarizing the metadata.
There may be incomplete sections in the metadata - this is expected, as we sometimes cannot obtain every piece of information about each dataset.

{{METADATA}}

<instructions>
Write a brief descriptive paragraph designed to provide an overview of this dataset.
The intended user is a researcher or scientist who needs to understand the context of that dataset in order to perform prediction or diagnostic tasks from the data. 
</instructions>

Rules:
<rules>
- Be precise and detailed.
- Do not use the word "machine learning" in the output
- Provide the description in plain language that would be appropriate for a person to understand the dataset even if they have never seen it before.
- Provide all details necessary for a scientist that might use the dataset for their own research or experiments.
- Describe all context needed for someone to understand the dataset.
- Information will likely be missing from the metadata. In these cases, give the most complete description you can based on the available information, and clearly state which factors are missing or not known.
- If possible, make sure to mention the class balance of the dataset (the balance of the target classes, and which is the majority class).
- Use this opportunity to highlight any relevant knowledge or details about which features or values in the dataset might be most predictive of the target. 
- Give a detailed description of any factors which might be related to the prediction target (either those present in the dataset, or not present in the dataset).
- Give a precise technical description of any context surrounding this dataset which might be relevant to understanding it; refer to specific information when providing context (i.e. specific events or facts related to the prediction target, population in the dataset, etc.), not general descriptions.
- Write the output as if it is describing one element from this dataset, and start it with a phrase that makes this clear, such as "This observation is drawn from a dataset..."
- Do not give any extra information beyond what is requested above or what would be useful to a researcher studying the specific prediction task described above.
- Do not wrap your output in any XML tags; only return plain text.
- IMPORTANT: Do not describe any other potential applications or uses of this dataset beyond for the prediction task described above. Do not describe why the dataset is valuable or useful except for the prediction task described above.
</rules>

Think about your answer first before you respond.

{{SUFFIX}}
"""

NUM_CONTEXT_VARIANTS = 8
# These instructions are the same as the task context instructions until the section after the {{METADATA}}.
# Additionally, it uses a superset of the rules in the task context instructions (the last few are added).
TASK_CONTEXT_VARIANTS_INSTRUCTIONS = f"""{{PREFIX}}
You are being asked to summarize a variety of metadata about a dataset for users who will utilize that data to perform a modeling task.
Below, after the task instructions, I will provide this metadata in a series of distinct sections. 
Each section starts with the characters '{TASK_SECTION_SEP.strip()}' and a short textual description of the metadata, before providing the metadata itself.

You are a helpful but concise, matter-of-fact, and scientific AI assistant, and should maintain a scientific tone.
Read the metadata below carefully, because I'm going to ask you to to produce output summarizing the metadata.

{{METADATA}}

<instructions>
Write {NUM_CONTEXT_VARIANTS} brief descriptive paragraphs. 
Each paragraph should be designed to provide an overview of this dataset.
The intended user is a researcher or scientist who needs to fully understand the dataset in order to perform modeling tasks using the data.
Each paragraph should have a different tone, style, and level of formality. 
</instructions>

Rules:
<rules>
- Be precise and detailed.
- Do not use the word "machine learning" in the output
- Provide the description in plain language that would be appropriate for a scientist to understand the dataset.
- If possible, make sure to mention the class balance of the dataset (the balance of the target classes, and which is the majority class).
- Use this opportunity to highlight any relevant knowledge or details about which features or values in the dataset might be most predictive of the target. 
- Write the output as if it is describing one element from this dataset, and start it with a phrase that makes this clear, such as "This observation is drawn from a dataset..."
- Do not give any extra information beyond what is requested above or what would be useful to a researcher studying the specific prediction task described above.
- Return the results as a JSON structure comprising the {NUM_CONTEXT_VARIANTS} examples. 
- The keys in the output JSON should be integers (0, 1, ..., {NUM_CONTEXT_VARIANTS}) and the values should be the string consiting of the dataset description you generated.
- Make sure every distinct description is different from the others. Be creative and vary the style and tone.
</rules>

<example>
For example, here is what ONE value in the corresponding output JSON might look like: 
{{EXAMPLE}}

Your output should be thorough and completely describe all aspects listed above. 
It should be similar in length and detail, but different in other characteristics (such as tone, structure, vocabulary, and the exact contents).
</example>

Think about your answer first before you respond. It is critical to the results of this scientific study that you follow the instructions above and return a valid JSON response - the scientists need our help to understand this data.

{{SUFFIX}}
"""

NUM_FEATURE_VARIANTS = 32
FEATURE_VARIANTS_EXAMPLE = f"""
{{
    0: "<new name here>", 
    1: "<another new name here>", 
    ..., 
    {NUM_FEATURE_VARIANTS}: "<final feature name>"
 }}"""

CODE_VARIANTS_EXAMPLE = """
{{
    "original_value_mapping_key_1": ["<new name here>", "<another new name", ...], 
    "original_value_mapping_key_2": ["<new name here>", "<another new name", ...], 
    ..., 
    "original_value_mapping_key_n": ["<new name here>", "<another new name", ...], 
 }}
 """

FEATURE_VARIANTS_INSTRUCTIONS = f"""{{PREFIX}}
You are being asked to generate a diverse set of names for a feature (or column) in a dataset to be used for a machine learning task.
Below, I will provide a variety of metadata regarding all of the features in this dataset, in a series of distinct sections. 
Each section starts with the characters '{TASK_SECTION_SEP.strip()}' and a short textual description of the metadata, before providing the metadata itself.

You are a helpful but concise, matter-of-fact, creative, and scientific AI assistant, and should maintain a scientific tone.
Read the metadata below carefully, because I'm going to ask you to to produce a diverse set of {NUM_FEATURE_VARIANTS} new names for the {{FEATURE}} feature in the dataset which preserve the original meaning.

{{METADATA}}

<info>
One critical component of the metadata above is the Python FeatureList object. 
The FeatureList is a data structure that contains a list of features (variables) describing a dataset. 
Each element of the FeatureList is a Feature object, which might include the following attributes:
- name: The name of the feature. This is always the first element of the Feature object.
- dtype: The data type (float, cat_dtype means categorical). This is always the second element of the feature object.
- value_mapping: The potential coded values that might occur for that feature, and the true values the codes represent
- name_extended: A more detailed name and description of the variable.
</info>

<input_feature>
For emphasis, here is the Feature, repeated from the FeatureList above, for the feature {{FEATURE}} which you are being asked to use:
{{SERIALIZED_FEATURE}}
</input_feature>

<instructions>
For the {{FEATURE}} feature, create a creative, varied, and diverse list of {NUM_FEATURE_VARIANTS} possible new descriptions of that feature. 
These should be approximately one sentence in length, but some longer descriptions are ok. 
- Use diverse wording, phrasing, and sentence structure for these descriptions. 
- All of the responses should be distinct and unique, and can use a different tone (i.e. some formal and scientific, some more informal; some lengthy and extensive, some short).
- The descriptions should preserve the meaning of the feature name and should still accurately describe what the feature is.
- Each description should be different from the others.
- Try using different tones (scientific, informal, tweet-style language), different spellings (i.e. American and British spellings), and changing the wording of the feature names.
- Always keep the meaning similar to the original feature name. 
</instructions>

<rules>
- Return the result as JSON object, mapping an integer to a string.
- The keys of the JSON should be integers, and the values should be strings.
- The top-level keys in the returned JSON should be a unique integer ID, counting from 0 to {NUM_FEATURE_VARIANTS - 1}.
- The values should be a string containing one of the {NUM_FEATURE_VARIANTS} new extended feature names you have created.
- Only return the JSON object. Do not return any other text, description, or output besides the JSON.
- Make sure there are exactly {NUM_FEATURE_VARIANTS} new names in the values of the returned JSON object, numbered 0 to {NUM_FEATURE_VARIANTS - 1}.
</rules>

<example>
For example, the structure of the output JSON should look like: 
{{FEATURE_VARIANTS_EXAMPLE}}
 
 The integers should count up from 0 to {NUM_FEATURE_VARIANTS}.
</example>

ONLY perform this task for the {{FEATURE}} feature. The other features are only provided for context.
Think about your answer first before you respond. Make sure that there are {NUM_FEATURE_VARIANTS} new names for each feature.

{{SUFFIX}}
"""

CODE_VARIANTS_INSTRUCTIONS = f"""{{PREFIX}}
You are being asked to generate a diverse set of names for each value of a feature (or column) in a dataset to be used for a machine learning task.
Below, I will provide a variety of metadata regarding all of the features in this dataset, in a series of distinct sections. 
Each section starts with the characters '{TASK_SECTION_SEP.strip()}' and a short textual description of the metadata, before providing the metadata itself.

You are a helpful but concise, matter-of-fact, creative, and scientific AI assistant, and should maintain a scientific tone.
Read the metadata below carefully, because I'm going to ask you to to produce a diverse set of {NUM_FEATURE_VARIANTS} new names for each value in the {{FEATURE}} feature in the dataset which preserve the original meaning.

{{METADATA}}

<info>
One critical component of the metadata above is the Python FeatureList object. 
The FeatureList is a data structure that contains a list of features (variables) describing a dataset. 
Each element of the FeatureList is a Feature object, which might include the following attributes:
- name: The name of the feature. This is always the first element of the Feature object.
- dtype: The data type (float, cat_dtype means categorical). This is always the second element of the feature object.
- value_mapping: The potential coded values that might occur for that feature, and the true values the codes represent. IMPORTANT: THE VALUE MAPPING IS WHAT YOU WILL NEED TO USE MOST FOR YOUR OUTPUTS.
- name_extended: A more detailed name and description of the variable.
</info>

<instructions>
For each entry in the value_mapping for the {{FEATURE}} feature, create a creative, varied, and diverse list of {NUM_FEATURE_VARIANTS} possible new mappings of that feature. 
These should match the original mappings in length.
- Use diverse wording, phrasing, and structure for these descriptions. 
- The descriptions should preserve the meaning of the mapping and should still accurately describe what that value is.
- All of the responses should be distinct and unique, and can use a different tone or style.
- Try using different tones (scientific, informal, tweet-style language), different spellings (i.e. American and British spellings), and changing the wording.
- Try using common abbreviations or codes for the values, if they exist.
- Each value should be different from the others.
- Always keep the keys in the value_mapping the same; you will only modify the values.
- Only generate responses in English. 
</instructions>

<rules>
- Return the result as JSON object. 
- The returned JSON should have the same keys as the original value_mapping for feature {{FEATURE}}.
- The value of the returned JSON should be a LIST of {NUM_FEATURE_VARIANTS} potential mappings for that key that you have created.
- Only return the JSON object. Do not return any other text, description, or output besides the JSON.
- Make sure there are exactly {NUM_FEATURE_VARIANTS} new names in the values of the returned JSON object.
</rules>

<example>
For example, the structure of the output JSON should look like: 
{{CODE_VARIANTS_EXAMPLE}}

 Each of the keys should be an array of length {NUM_FEATURE_VARIANTS}.
</example>

ONLY perform this task for the {{FEATURE}} feature. The other features are only provided for context.
Think about your answer first before you respond. 
Make sure that there are {NUM_FEATURE_VARIANTS} new values for each entry in your returned JSON. 
Count the values in each array one by one while generating your JSON response.

{{SUFFIX}}
"""
