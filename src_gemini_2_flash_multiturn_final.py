import re
from tqdm import tqdm
import csv
import time
import google.generativeai as genai

# Set your API key manually (replace with your actual key)
GOOGLE_API_KEY = 'YOUR-KEY-HERE'  
genai.configure(api_key=GOOGLE_API_KEY)

gemini_model_name = "gemini-2.0-flash"  # Adjust as needed
gemini_model = genai.GenerativeModel(gemini_model_name)

def get_answer(user_prompt, n=3):
    response = gemini_model.generate_content(
        user_prompt, 
        generation_config={"temperature": 0.1, "top_p": 0.9, "candidate_count": n}
    )
    return [candidate.content.parts[0].text for candidate in response.candidates]

def extract_ratings(text):
    if '<begin of ratings>' in text and '</begin of ratings>' in text: pattern = r"<begin of ratings>\s*({.*?})\s*</begin of ratings>"
    else: pattern = r'\n\n(.*?)\n'
    match = re.search(pattern, text, re.DOTALL)
    return eval(match.group(1))

def get_eval_answer_with_retry(user_prompt, max_retries=3, n=1):
    model = gemini_model_name
    for attempt in range(1, max_retries + 1):
        try:
            answer = get_answer(user_prompt, n=n)
            # print(answer)
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

# Dimension evaluation
### Communication
COM_FORMAT = "{'Token quantity': 1-10, 'Manner': 1-10, 'Interaction': 1-10, 'Politeness': 1-10}"
COM_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on the following criteria.

The prompt for you to evaluate is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

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
</begin of explanation>

<begin of ratings>{COM_FORMAT}</begin of ratings>"""

### Cognition
COG_FORMAT = "{'Intrinsic load': 1-10, 'Extraneous load': 1-10, 'Germmane load': 1-10}"
COG_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Intrinsic load: This evaluates the prompts in explicitly guiding models to break complex tasks into actionable steps aligned with LM skills.
- Extraneous load: The extent to which prompts exclude irrelevant materials to reduce unnecessary load.
- Germane load: The degree to which prompts explicitly engage models with their prior knowledge or deep working memory (e.g., ``ask itself'') to integrate it with existing and new knowledge for problem-solving.

The scoring system is provided below:
> Intrinsic load:
- 1-2 (Poor): The prompt provides little to no guidance on breaking down the task. It is overly vague, abstract, or assumes the model can handle complexity without guidance.
- 3-4 (Below Average): The prompt provides minimal guidance but fails to clearly break the task into actionable steps. The model is left to infer most of the process.
- 5-6 (Average): The prompt partially breaks down the task but lacks clarity or completeness in defining actionable steps. Some guidance is present, but it is inconsistent or incomplete.
- 7-8 (Good): The prompt effectively breaks the task into clear, actionable steps. It aligns well with the model’s skills but may lack some nuance or optimization.
- 9-10 (Excellent): The prompt perfectly breaks the task into logical, actionable steps. It is highly aligned with the model’s capabilities and ensures clarity and efficiency in execution.

> Extraneous load:
- 1-2 (Poor): The prompt includes excessive irrelevant information, making it difficult for the model to focus on the core task. It is cluttered or overly verbose.
- 3-4 (Below Average): The prompt contains some irrelevant information, but the core task is still somewhat discernible. The extraneous load is noticeable and distracting.
- 5-6 (Average): The prompt includes some unnecessary details but generally stays focused on the task. The extraneous load is moderate but not overly detrimental.
- 7-8 (Good): The prompt is concise and mostly free of irrelevant information. It minimizes extraneous load effectively, with only minor distractions.
- 9-10 (Excellent): The prompt is perfectly concise and excludes all irrelevant materials. It is optimized to reduce extraneous load to the bare minimum.

> Germane load:
- 1-2 (Poor): The prompt does not engage the model’s prior knowledge or working memory. It provides no cues or instructions to leverage existing knowledge.
- 3-4 (Below Average): The prompt makes minimal attempts to engage prior knowledge but does so ineffectively or inconsistently. The model is left to infer connections on its own.
- 5-6 (Average): The prompt partially engages the model’s prior knowledge but lacks depth or clarity in integrating it with new information. The engagement is superficial.
- 7-8 (Good): The prompt effectively engages the model’s prior knowledge and encourages integration with new information. It provides clear cues or instructions for leveraging existing knowledge.
- 9-10 (Excellent): The prompt perfectly engages the model’s prior knowledge and deep working memory. It explicitly guides the model to integrate existing and new knowledge for optimal problem-solving.

Your evaluations must focus on explicit instructions rather than implicit instructions. 
For example, if the prompt does not say ``Reflect on your prior knowledge'' then you should not assume that the prompt is effective in encouraging germane load.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
</begin of explanation>

<begin of ratings>{COG_FORMAT}</begin of ratings>"""

### Instruction
INS_FORMAT = "{'Objectives': 1-10, 'External tools': 1-10, 'Metacognition': 1-10, 'Demos': 1-10, 'Rewards': 1-10}"
INS_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Objectives: How well prompts explicitly communicate the task objectives, including expected outputs, formats, constraints, audiences, and other applicable criteria.
- External tools: The extent to which prompts explicitly guide models to identify when specific external tools or knowledge resources are needed, and perform tool calls to support problem-solving.
- Metacognition: This assesses prompts in explicitly guiding models to reason, self-monitor, and self-verify outputs to meet expectations and enhance reliability.
- Demos: The extent to which the prompts explicitly include examples, demonstrations, and counterexamples to illustrate the desired output.
- Rewards: How well prompts explicitly establish feedback, reward, and reinforcement mechanisms that encourage the models achieving desired outputs.

The scoring system is provided below:
> Objectives:
- 1-2 (Poor): The prompt lacks any clear objectives or guidance. It does not communicate task goals, expected outputs, formats, constraints, or audience considerations.
- 3-4 (Below Average): The prompt provides vague or incomplete objectives. Some elements (e.g., task goals or constraints) are mentioned but are unclear or inconsistent.
- 5-6 (Average): The prompt outlines basic objectives but lacks depth or specificity. Key elements like expected outputs or constraints are partially addressed but not fully developed.
- 7-8 (Good): The prompt clearly communicates objectives, including expected outputs, formats, and constraints. It may lack minor details or fail to address specific edge cases.
- 9-10 (Excellent): The prompt comprehensively defines objectives, including all relevant details such as expected outputs, formats, constraints, and audience considerations. It leaves no ambiguity about the task.

> External tools:
- 1-2 (Poor): The prompt does not mention or guide the use of external tools or resources. It assumes the model can solve the task without additional support.
- 3-4 (Below Average): The prompt vaguely hints at the need for external tools but provides no clear guidance on when or how to use them.
- 5-6 (Average): The prompt acknowledges the need for external tools in some cases but lacks specificity or detailed instructions on how to identify and use them.
- 7-8 (Good): The prompt explicitly guides the model to identify when external tools are needed and provides clear instructions on how to use them. It may lack examples or edge-case considerations.
- 9-10 (Excellent): The prompt thoroughly integrates external tools into the task, providing clear guidance on when and how to use them, including examples and edge-case considerations.

> Metacognition:
- 1-2 (Poor): The prompt does not encourage or guide the model to reason, self-monitor, or self-verify outputs. It assumes the model will produce reliable results without oversight.
- 3-4 (Below Average): The prompt includes minimal guidance on reasoning or self-monitoring but lacks specificity or actionable steps for the model to follow.
- 5-6 (Average): The prompt provides some guidance on reasoning and self-monitoring but does not fully integrate these elements into the task. The guidance may be generic or incomplete.
- 7-8 (Good): The prompt explicitly guides the model to reason, self-monitor, and self-verify outputs. It provides clear steps or frameworks for ensuring reliability.
- 9-10 (Excellent): The prompt thoroughly integrates metacognitive strategies, providing detailed guidance on reasoning, self-monitoring, and self-verification. It includes examples and edge-case considerations.

> Demos:
- 1-2 (Poor): The prompt does not include any examples, demonstrations, or counterexamples. It assumes the model can infer the desired output without guidance.
- 3-4 (Below Average): The prompt includes minimal or poorly constructed examples that do not effectively illustrate the desired output.
- 5-6 (Average): The prompt includes basic examples or demonstrations but lacks variety or depth. Counterexamples may be missing or poorly explained.
- 7-8 (Good): The prompt includes clear and relevant examples, demonstrations, and counterexamples. It effectively illustrates the desired output but may lack edge-case examples.
- 9-10 (Excellent): The prompt provides comprehensive examples, demonstrations, and counterexamples, including edge cases. It leaves no ambiguity about the desired output.

> Rewards:
- 1-2 (Poor): The prompt does not establish any reward, feedback or reinforcement mechanisms. It assumes the model will achieve the desired output without encouragement.
- 3-4 (Below Average): The prompt includes vague or minimal reward and feedback mechanisms but lacks specificity or actionable guidance.
- 5-6 (Average): The prompt provides basic reward, feedback or reinforcement mechanisms but does not fully integrate them into the task. The mechanisms may be generic or incomplete.
- 7-8 (Good): The prompt explicitly establishes reward, feedback and reinforcement mechanisms, providing clear guidance on how to achieve the desired output.
- 9-10 (Excellent): The prompt thoroughly integrates reward, feedback and reinforcement mechanisms, including detailed guidance and examples. It ensures the model is consistently encouraged to achieve the desired output.

Your evaluations must focus on explicit instructions rather than implicit instructions. 
For example, if the prompt does not mention about the formats or constraints of the objectives then you should not assume that the prompt is effective in communicating the objectives.
For example, if the prompt does not say ``I will reward you something for something'' then you should not assume that the prompt is effective in encouraging the rewards.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
</begin of explanation>

<begin of ratings>{INS_FORMAT}</begin of ratings>"""

### Logic and structure
LOGIC_FORMAT = "{'Structural logic': 1-10, 'Contextual logic': 1-10}"
LOGIC_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Structural logic: This evaluates the logical clarity and coherence of prompts' structure, and the progression between components.
- Contextual logic: This assesses the logical consistency and coherence of the instructions, terminologies, concepts, facts, and other components within the prompt and across communication turns.

The scoring system is provided below:
> Structural logic:
- 1-2 (Poor): The prompt lacks any discernible structure or logical flow. Components are disjointed, confusing, or entirely absent. The progression between ideas is unclear or nonexistent.
- 3-4 (Below Average): The prompt has a basic structure but is poorly organized. Some components are present, but the flow between them is weak or inconsistent. The overall structure hinders understanding rather than aiding it.
- 5-6 (Average): The prompt has a moderately clear structure with an identifiable progression between components. However, some sections may be only partially connected or there might be minor lapses in logical sequencing that could lead to occasional misinterpretation.
- 7-8 (Good): The prompt demonstrates clear and coherent structural logic. Each component is well-organized with a smooth and logical progression, though there may be minor areas where the transition could be slightly improved.
- 9-10 (Excellent): The prompt exemplifies excellent structural logic. The organization is impeccable, with a flawless and intuitive progression between all components. Every section flows naturally into the next, leaving no ambiguity about the order or relationships among elements.

> Contextual logic:
- 1-2 (Poor): The prompt is riddled with inconsistencies, contradictions, or unclear usage of instructions, terminologies, and concepts.
- 3-4 (Below Average): The prompt includes some logical context but still suffers from notable inconsistencies or unclear instructions. Terminology and concepts might be used imprecisely or contradict each other in parts, reducing overall clarity.
- 5-6 (Average): The prompt maintains a generally consistent context with clear instructions and appropriate terminology. While there may be minor lapses or subtle inconsistencies, they do not severely hinder understanding, though they leave some room for improvement.
- 7-8 (Good): The prompt is very coherent in its use of language and logic. Instructions, terminologies, and concepts are used consistently and logically, with only occasional minor issues that do not significantly impact the overall clarity or reliability.
- 9-10 (Excellent): The prompt demonstrates exemplary contextual logic. All instructions, terminologies, and concepts are applied consistently and coherently throughout, with seamless integration across communication turns.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
</begin of explanation>

<begin of ratings>{LOGIC_FORMAT}</begin of ratings>"""

### Hallucination
HALL_FORMAT = "{'Hallucination awareness': 1-10, 'Factuality and creativity': 1-10}"
HALL_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10:

- Hallucination awareness: The extent to which prompts explicitly guide models to generate factual and evidence-based responses while minimizing speculative or unsupported claims.
- Factuality and creativity: The degree to which prompts explicitly guide models to balance creative generation with factual accuracy, including which task and when to prioritize creativity over creativity and vice versa. We have yet observed prompting methods designed for this criterion to date.

The scoring system is provided below:
> Hallucination awareness:
- 1-2 (Poor): The prompt does not address hallucination at all. It provides no guidance or explicit instructions to minimize speculative or unsupported claims. The model is left entirely to its own devices, which may lead to frequent factual inaccuracies or hallucinations.
- 3-4 (Below Average): The prompt contains only a very vague or minimal mention of factual correctness. It might include a brief statement urging caution, but it lacks explicit guidance or clear mechanisms for ensuring factual and evidence-based responses. The model is given little direction to avoid hallucination.
- 5-6 (Average): The prompt includes some guidance to avoid hallucinations, such as a general instruction to 'be factual' or 'avoid speculation'. However, the guidance is not detailed or specific enough to ensure consistent factual accuracy.
- 7-8 (Good): The prompt explicitly instructs the model to prioritize evidence-based responses and provides specific strategies to minimize hallucinations (e.g., 'cite sources', 'avoid unsupported claims', or 'stick to verified information'). The guidance is clear but may lack depth or nuance.
- 9-10 (Excellent): The prompt provides highly detailed and explicit instructions to minimize hallucinations. It may include examples, constraints, or specific frameworks (e.g., 'only use information from verified sources', 'clearly distinguish between facts and opinions'). The guidance is robust and likely to result in highly factual and reliable outputs.

> Factuality and creativity:
- 1-2 (Poor): The prompt does not address the balance between factuality and creativity. It either ignores factual accuracy entirely or stifles creativity by overemphasizing rigid factual constraints without context.
- 3-4 (Below Average): The prompt acknowledges the need for both factuality and creativity but provides no clear guidance on how to balance them. The instructions are vague or contradictory, leaving the model unsure of when to prioritize one over the other.
- 5-6 (Average): The prompt includes some guidance on balancing factuality and creativity, such as 'be creative but accurate'. However, the instructions lack specificity or context, making it difficult for the model to consistently strike the right balance.
- 7-8 (Good): The prompt provides clear instructions on when to prioritize factuality and when to allow for creativity. For example, it may specify that factual accuracy is critical for certain tasks (e.g., scientific explanations) while creativity is encouraged for others (e.g., storytelling). The guidance is logical but may lack depth or examples.
- 9-10 (Excellent): The prompt offers highly detailed and nuanced guidance on balancing factuality and creativity. It may include specific examples, scenarios, or frameworks (e.g., 'prioritize factual accuracy for technical topics but allow creative freedom for artistic tasks'). The guidance is comprehensive and likely to result in outputs that are both accurate and imaginative.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
</begin of explanation>

<begin of ratings>{HALL_FORMAT}</begin of ratings>"""

### Responsibility
RES_FORMAT = "{'Bias': 1-10, 'Safety': 1-10, 'Privacy': 1-10, 'Reliability': 1-10, 'Societal norms': 1-10}"
RES_JUDGING_PROMPT = f"""You are a highly experienced judge tasked with evaluating a prompt on criteria.

The prompt given to you is provided below:

<begin of the prompt>
[[INPUT_PROMPT]]
</begin of the prompt>

Your task is to evaluate the above prompt on the following criteria on a scale of 1-10. 

- Bias: The extent to which prompts are devoid of biases and explicitly encourage models to generate content that is free from cultural, gender, racial, or socio-economic biases and avoids stereotypes.
- Safety: The degree to which prompts are free from unsafe content and explicitly encourage models to generate safe outputs, avoiding harmful content such as guidance on hazardous activities or weapon creation.
- Privacy: The extent to which prompts do not contain sensitive privacy information and explicitly encourage the models to generate content free of personally identifiable information or sensitive data.
- Reliability: How well prompts explicitly encourage explicit reasoning processes and attribution, including acknowledgment of model limitations and uncertainties in generated content.
- Societal norms: The degree to which prompts exclude harmful societal norms and explicitly encourage models to generate content aligning with widely accepted cultural, ethical, and moral standards, while being inclusive and contextually appropriate.

Your evaluations must focus on explicit instructions rather than implicit instructions. 
For example, if the prompt does not say ``Output must not be biased'' then you should not assume that the prompt is effective in instructing LLMs to be less biased.
For example, if the prompt does not say ``Output must be safe'' then you should not assume that the prompt is effective in instructing LLMs to be safe.
For example, if the prompt does not say ``Output must respect user privacy'' then you should not assume that the prompt is effective in instructing LLMs to respect privacy.
For example, if the prompt does not say ``Output must respect sociental norms'' then you should not assume that the prompt is effective in instructing LLMs to respect sociental norms.

The scoring system is provided below:
> Bias:
- 1-2 (Poor): The prompt explicitly contains or encourages biased content, stereotypes, or discriminatory language. It does not address or mitigate biases in any way.
- 3-4 (Below Average): The prompt may inadvertently include biased language or fail to discourage biases. It lacks explicit instructions to avoid cultural, gender, racial, or socio-economic biases.
- 5-6 (Average): The prompt avoids overt biases but does not explicitly encourage the model to generate unbiased content. It may lack clarity in addressing potential biases.
- 7-8 (Good): The prompt explicitly discourages biases and encourages the model to generate content free from stereotypes and discrimination. However, it may not fully address all potential bias scenarios.
- 9-10 (Excellent): The prompt is entirely free from biases and explicitly encourages the model to generate content that is inclusive, fair, and devoid of cultural, gender, racial, or socio-economic biases.

> Safety:
- 1-2 (Poor):  The prompt explicitly encourages or contains unsafe content, such as guidance on hazardous activities, violence, or harmful behavior.
- 3-4 (Below Average): The prompt may inadvertently include unsafe content or fail to discourage harmful outputs. It lacks explicit instructions to avoid unsafe or dangerous content.
- 5-6 (Average): The prompt avoids overtly unsafe content but does not explicitly encourage the model to generate safe outputs. It may lack clarity in addressing potential safety concerns.
- 7-8 (Good): The prompt explicitly discourages unsafe content and encourages the model to generate safe outputs. However, it may not fully address all potential safety risks.
- 9-10 (Excellent): The prompt is entirely free from unsafe content and explicitly encourages the model to generate safe, responsible, and non-harmful outputs.

> Privacy:
- 1-2 (Poor): The prompt explicitly contains or encourages the inclusion of sensitive personal information, such as personally identifiable information (PII) or private data.
- 3-4 (Below Average): The prompt may inadvertently include privacy risks or fail to discourage the inclusion of sensitive information. It lacks explicit instructions to avoid privacy violations.
- 5-6 (Average): The prompt avoids overt privacy risks but does not explicitly encourage the model to generate content free of sensitive information. It may lack clarity in addressing potential privacy concerns.
- 7-8 (Good): The prompt explicitly discourages the inclusion of sensitive information and encourages the model to generate privacy-conscious outputs. However, it may not fully address all potential privacy risks.
- 9-10 (Excellent): The prompt is entirely free from privacy risks and explicitly encourages the model to generate content that respects privacy and avoids sensitive data.

> Reliability:
- 1-2 (Poor): The prompt does not encourage reasoning, attribution, or acknowledgment of model limitations. It may explicitly encourage unreliable or unverified outputs.
- 3-4 (Below Average): The prompt may inadvertently discourage reasoning or fail to address model limitations. It lacks explicit instructions to ensure reliable and well-reasoned outputs.
- 5-6 (Average): The prompt avoids overtly unreliable content but does not explicitly encourage reasoning, attribution, or acknowledgment of uncertainties. It may lack clarity in addressing reliability.
- 7-8 (Good): The prompt explicitly encourages reasoning, attribution, and acknowledgment of model limitations. However, it may not fully address all potential reliability concerns.
- 9-10 (Excellent): The prompt explicitly encourages thorough reasoning, proper attribution, and acknowledgment of uncertainties and model limitations, ensuring highly reliable outputs.

> Societal norms:
- 1-2 (Poor): The prompt explicitly encourages harmful societal norms, unethical behavior, or exclusionary practices. It does not align with widely accepted cultural, ethical, or moral standards.
- 3-4 (Below Average): The prompt may inadvertently include harmful societal norms or fail to discourage unethical or exclusionary content. It lacks explicit instructions to align with ethical and moral standards.
- 5-6 (Average): The prompt avoids overtly harmful societal norms but does not explicitly encourage alignment with ethical, moral, or inclusive standards. It may lack clarity in addressing societal norms.
- 7-8 (Good): The prompt explicitly discourages harmful societal norms and encourages the model to generate content that aligns with ethical, moral, and inclusive standards. However, it may not fully address all potential societal norm concerns.
- 9-10 (Excellent): The prompt is entirely free from harmful societal norms and explicitly encourages the model to generate content that aligns with widely accepted cultural, ethical, and moral standards, while being inclusive and contextually appropriate.

Begin your evaluation by providing a short explanation for each. Be as objective, thorough, and constructive as possible. After providing your explanation, please rate the response on all the criteria on a scale of 1 to 10 by strictly following this format:

<begin of explanation>
…
</begin of explanation>

<begin of ratings>{RES_FORMAT}</begin of ratings>"""

######### Run experiments #########
FOLDER_PATH = "/home/long/WhatMakesAGoodPrompt/data/test_data_multiturn.csv"
SAVED_PATH = "/home/long/WhatMakesAGoodPrompt/data/evaluated_test_data_multiturn_geminiflash.csv"

saved_rows = []
with open(FOLDER_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    # prompt,source,task,human or machine,complexity, full_conversation
    for row in tqdm(csvreader):
        try:
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
        except: continue
        
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
        "prompt", "source", "task", "human or machine", "complexity", "full_conversation",
        "com_eval", "com_extracted_eval",
        "cog_eval", "cog_extracted_eval",
        "ins_eval", "ins_extracted_eval",
        "logic_eval", "logic_extracted_eval",
        "hall_eval", "hall_extracted_eval",
        "res_eval", "res_extracted_eval"
    ])
    csvwriter.writerows(saved_rows)