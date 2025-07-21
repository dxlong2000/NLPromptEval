import openai
import re
from tqdm import tqdm
import csv
import time

WING_KEY = "YOUR-KEY-HERE"

def get_gpt4o_mini_answer(user_prompt, max_tokens=2048, temperature=0.1, system_prompt="You are a helpful assistant!", n=1):
    api_token = WING_KEY
    model_name = "gpt-4o-mini"
    openai.api_key = api_token
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n
    )
    if n==1: return response.choices[0].message.content
    else: return [response.choices[idx].message.content for idx in range(n)]

def get_gpt4o_answer(user_prompt, max_tokens=2048, temperature=0.1, system_prompt="You are a helpful assistant!", n=1):
    api_token = WING_KEY
    model_name = "gpt-4o-2024-11-20"
    openai.api_key = api_token
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n
    )
    if n==1: return response.choices[0].message.content
    else: return [response.choices[idx].message.content for idx in range(n)]

def get_answer(user_prompt, model="gpt-4o", n=1):
    if model == "gpt-4o": return get_gpt4o_answer(user_prompt, n=n)
    elif model == "gpt-4o-mini": return get_gpt4o_mini_answer(user_prompt, n=n)

def extract_ratings(text):
    pattern = r"<begin of ratings>\s*({.*?})\s*<end of ratings>"
    match = re.search(pattern, text, re.DOTALL)
    return eval(match.group(1))

def get_eval_answer_with_retry(user_prompt, model="gpt-4o", max_retries=3, n=1):
    for attempt in range(1, max_retries + 1):
        try:
            answer = get_answer(user_prompt, model=model, n=n)
            if n == 1:
                extracted_ans = extract_ratings(answer)
                return answer, extracted_ans
            else:
                extracted_ans = [extract_ratings(ans) for ans in answer]
                return answer, extracted_ans
        except Exception as e:
            print(f"Error in {model} answer (attempt {attempt}/{max_retries}): {str(e)}")
            time.sleep(3)
    raise Exception(f"Max retries reached. Unable to get a valid response from {model}.")

######### Run experiments #########
# Load checklist questions
FOLDER_PATH = "/home/ComplexPrompt/data/test_data.csv"
SAVED_PATH = "/home/ComplexPrompt/src/properties-eval/evaluated_test_data_enhanced.csv"

# Dimension evaluation
### Communication
COM_FORMAT = "{'Token quantity': 1-10, 'Manner': 1-10, 'Interaction': 1-10, 'Politeness': 1-10}"
COM_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on the following criteria.

The prompt for you to evaluate is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria and rate each criterion on a scale of 1-10:

- Token quantity: The extent to which prompts provide optimal and relevant information while minimizing token usage, balancing information completeness with efficiency.
- Manner: The degree to which prompt is clear and direct (across turns) while minimizing unnecessary ambiguity, complexity, and confusion.
- Interaction: The extent to which the prompts explicitly encourage the models to gather the necessary details and requirements by asking questions of clarification or confirmation.
- Politeness: The degree to which prompt maintains professional and context-specific politeness.

The scoring system is provided below:
> Token quantity:
- 1-2 (Poor): The prompt is highly inefficient with token usage. It includes excessive, redundant details or is overly wordy without adding meaningful information. It either lacks critical information or includes irrelevant details, making it difficult for the model to understand or respond effectively.
- 3-4 (Below Average): The prompt is either too long or too short, with noticeable inefficiencies in token usage. It may include some unnecessary information or omit key details, reducing its effectiveness.
- 5-6 (Average): The prompt is moderately efficient in token usage but could be improved. It includes most necessary information but may have minor redundancies or omissions.
- 7-8 (Good): The prompt is efficient in token usage, providing a good balance between information completeness and conciseness. It includes all necessary details without significant redundancy.
- 9-10 (Excellent): The prompt is highly efficient in token usage, providing optimal and relevant information with minimal redundancy. It is concise yet comprehensive, enabling the model to respond effectively.

> Manner:
- 1-2 (Poor): The prompt is unclear, ambiguous, or overly complex, leading to significant confusion. It lacks directness and may require multiple interpretations.
- 3-4 (Below Average): The prompt has noticeable issues with clarity or directness. It may contain unnecessary complexity or ambiguity, making it harder for the model to understand.
- 5-6 (Average): The prompt is generally clear but could be more direct or simplified. It may have minor ambiguities or complexities that do not severely hinder understanding.
- 7-8 (Good): The prompt is clear and direct, with minimal ambiguity or complexity. It is easy for the model to understand and respond to.
- 9-10 (Excellent): The prompt is exceptionally clear, direct, and free of ambiguity or complexity. It is straightforward and easy for the model to interpret.

> Interaction
- 1-2 (Poor): The prompt does not encourage interaction or clarification. It assumes all necessary information is provided and does not prompt the model to ask questions.
- 3-4 (Below Average): The prompt minimally encourages interaction but lacks explicit guidance for the model to ask clarifying or confirming questions.
- 5-6 (Average): The prompt somewhat encourages interaction but could be more explicit in guiding the model to ask questions or seek clarification.
- 7-8 (Good): The prompt effectively encourages interaction, explicitly guiding the model to ask clarifying or confirming questions when necessary.
- 9-10 (Excellent): The prompt excellently encourages interaction, clearly and explicitly prompting the model to gather all necessary details through questions or confirmation.

> Politeness
- 1-2 (Poor): The prompt is unprofessional, impolite, or inappropriate for the context. It may use offensive or overly casual language.
- 3-4 (Below Average): The prompt lacks consistent politeness or professionalism. It may have moments of appropriateness but fails to maintain a respectful tone throughout.
- 5-6 (Average): The prompt is generally polite and professional but could be more consistent or context-specific in its tone.
- 7-8 (Good): The prompt maintains a professional and polite tone throughout, with minor room for improvement in context-specificity.
- 9-10 (Excellent): The prompt is exceptionally polite, professional, and context-specific. It maintains a respectful and appropriate tone at all times.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{COM_FORMAT}
<end of ratings>"""

### Cognition
COG_FORMAT = "{'Intrinsic load': 1-10, 'Extraneous load': 1-10, 'Germmane': 1-10}"
COG_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Intrinsic load: This evaluates the prompts in explicitly guiding models to break complex tasks into actionable steps aligned with LM skills.
- Extraneous load: The extent to which prompts exclude irrelevant materials to reduce unnecessary load.
- Germane load: The degree to which prompts explicitly engage models with their prior knowledge or deep working memory (e.g., ``ask itself'') to integrate it with existing and new knowledge for problem-solving.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{COG_FORMAT}
<end of ratings>"""

### Instruction
INS_FORMAT = "{'Objectives': 1-10, 'External tools': 1-10, 'Metacognition': 1-10, 'Demos': 1-10, 'Rewards': 1-10}"
INS_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Objectives: How well prompts communicate the task objectives, including expected outputs, formats, constraints, audiences, and other applicable criteria.
- External tools: The extent to which prompts guide models to identify when specific external tools or knowledge resources are needed, and perform tool calls to support problem-solving.
- Metacognition: This assesses prompts in guiding models to reason, self-monitor, and self-verify outputs to meet expectations and enhance reliability.
- Demos: The extent to which the prompts include examples, demonstrations, and counterexamples to illustrate the desired output.
- Rewards: How well prompts establish feedback and reinforcement mechanisms that encourage the models achieving desired outputs.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{INS_FORMAT}
<end of ratings>"""

### Logic and structure
LOGIC_FORMAT = "{'Structural logic': 1-10, 'Contextual logic': 1-10}"
LOGIC_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Structural logic: This evaluates the logical clarity and coherence of prompts' structure, and the progression between components.
- Contextual logic: This assesses the logical consistency and coherence of the instructions, terminologies, concepts, facts, and other components within the prompt and across communication turns.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{LOGIC_FORMAT}
<end of ratings>"""

### Hallucination
HALL_FORMAT = "{'Hallucination awareness': 1-10, 'Factuality and creativity': 1-10}"
HALL_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Hallucination awareness: The extent to which prompts guide models to generate factual and evidence-based responses while minimizing speculative or unsupported claims.
- Factuality and creativity: The degree to which prompts guide models to balance creative generation with factual accuracy, including which task and when to prioritize creativity over strict adherence to factual content.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{HALL_FORMAT}
<end of ratings>"""

### Responsibility
RES_FORMAT = "{'Bias': 1-10, 'Safety': 1-10, 'Privacy': 1-10, 'Reliability': 1-10, 'Societal norms': 1-10}"
RES_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
<end of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Bias: The extent to which prompts are devoid of biases, and encourage models to generate content that is free from cultural, gender, racial, or socio-economic biases and avoids stereotypes.
- Safety: The degree to which prompts are free from unsafe contents, and safeguard LMs from generating harmful content such as guidance on hazardous activities or weapon creation.
- Privacy: The extent to which prompts do not contain privacy-sensitive information and invoke the generation of personally identifiable information or sensitive data, whether explicitly or implicitly.
- Reliability: How well prompts encourage explicit reasoning processes, feedback, and attribution, including acknowledgment of model limitations and uncertainties in generated content.
- Societal norms: The degree to which prompts exclude negative societal norms and instruct LMs to respect widely accepted cultural, ethical, and moral standards, encouraging generated content to be respectful, inclusive, and contextually appropriate.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
<end of explanation>

<begin of ratings>
{RES_FORMAT}
<end of ratings>"""

saved_rows = []
with open(FOLDER_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    # prompt,source,task,human or machine,complexity
    for row in tqdm(csvreader):
        prompt = row[0]
        
        com_prompt = COM_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        com_eval, com_extracted_eval = get_eval_answer_with_retry(com_prompt, n=3)
        
        cog_prompt = COG_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        cog_eval, cog_extracted_eval = get_eval_answer_with_retry(cog_prompt, n=3)
        
        ins_prompt = INS_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        ins_eval, ins_extracted_eval = get_eval_answer_with_retry(ins_prompt, n=3)
        
        logic_prompt = LOGIC_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        logic_eval, logic_extracted_eval = get_eval_answer_with_retry(logic_prompt, n=3)
        
        hall_prompt = HALL_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        hall_eval, hall_extracted_eval = get_eval_answer_with_retry(hall_prompt, n=3)
        
        res_prompt = RES_JUDGING_PROMPT.replace("[[INPUT_PROMPT]]", prompt)
        res_eval, res_extracted_eval = get_eval_answer_with_retry(res_prompt, n=3)
        
        tmp_row = row
        tmp_row.extend([
            com_eval, com_extracted_eval,
            cog_eval, cog_extracted_eval,
            ins_eval, ins_extracted_eval,
            logic_eval, logic_extracted_eval,
            hall_eval, hall_extracted_eval,
            res_eval, res_extracted_eval
        ])
        saved_rows.append(tmp_row)        

with open(SAVED_PATH, "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "prompt", "source", "task", "human or machine", "complexity",
        "com_eval", "com_extracted_eval",
        "cog_eval", "cog_extracted_eval",
        "ins_eval", "ins_extracted_eval",
        "logic_eval", "logic_extracted_eval",
        "hall_eval", "hall_extracted_eval",
        "res_eval", "res_extracted_eval"
    ])
    csvwriter.writerows(saved_rows)