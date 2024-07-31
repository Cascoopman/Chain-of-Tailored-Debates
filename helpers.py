import openai
import os
from typing import Tuple
import random

# Utilities

api_key = os.environ.get("OPENAI_API_KEY")
client_local = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)
client_openai = openai.OpenAI(
    api_key=api_key,
)

# Functions for LLM invocation  
  
def phi3_response(messages: list) -> str:
    response = client_local.chat.completions.create(
        model="phi3:14b-instruct",
        temperature=0.0,
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content

def gpt4o_mini_response(messages: list) -> str:
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content

def gpt4o_response(messages: list) -> str:
    response = client_openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        n=1,
        messages=messages,
    )

    return response.choices[0].message.content

# dict that converts string into function response
response_dict = {"gpt4o": gpt4o_response,
                 "gpt4o_mini": gpt4o_mini_response,
                 "phi3": phi3_response,}

# Functions for sentence and entity extraction

def create_summary_sentence_extractor_messages(summary: str) -> list:
    '''This function initiates the messages for invoking an LLM response that extracts sentences from a summary.
    The conversation list are messages including the system prompt, 3-shot prompting and the summary.

    - Input: a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": '''You are an expert at sentence extraction. It is your task to separate a summary into sentences. Make sure to place '\n' between sentences.'''},
                {"role": "user", "content": '''Summary: Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports. Journalists at Bild and Paris Match are "very confident" the video clip is real, an editor says. Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says.'''},
                {"role": "assistant", "content": '''Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports.\nJournalists at Bild and Paris Match are "very confident" the video clip is real, an editor says.\nAndreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says.'''},
                {"role": "user", "content": '''Summary: Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June. Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis.'''},
                {"role": "assistant", "content": '''Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June.\nIsrael and the United States opposed the move, which could open the door to war crimes investigations against Israelis.'''},
                {"role": "user", "content": '''Summary: Amnesty's annual death penalty report catalogs encouraging signs, but setbacks in numbers of those sentenced to death. Organization claims that governments around the world are using the threat of terrorism to advance executions. The number of executions worldwide has gone down by almost 22% compared with 2013, but death sentences up by 28%.'''},
                {"role": "assistant", "content": '''Amnesty's annual death penalty report catalogs encouraging signs, but setbacks in numbers of those sentenced to death.\nOrganization claims that governments around the world are using the threat of terrorism to advance executions.\nThe number of executions worldwide has gone down by almost 22% compared with 2013, but death sentences up by 28%.'''},
                {"role": "user", "content": f'''Summary: {summary}'''}
            ]

def create_sentence_statement_extractor_messages(summary: str, sentence: str) -> list:
    '''This function initiates the messages for invoking an LLM response that extracts sentences from a summary.
    The conversation list are messages including the system prompt, 3-shot prompting and the summary.

    - Input: a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert at statement extraction. 
                    
                    You are tasked with breaking down a sentence into smaller components, such that each statement can be individually fact-checked. 
                    Ensure that each statement is a standalone fact or piece of information, such that each can be individually verified or fact-checked.
                        
                    Given a summary and a highlighted sentence, accurately extract the statements from the highlighted sentence.    
                    Make sure to place '\n' between sentences.
                    '''},
                {"role": "user", "content": 
                    f'''Summary: {summary}
                    
                    Highlighted sentence: [{sentence}]
                    
                    Statements:
                    '''}
            ]

def create_entity_extractor_messages(sentence: str) -> list:
    '''This function initiates the messages for invoking an LLM response that extracts the entities from a sentence.
    The conversation list are messages including the system prompt, 3-shot prompting and the sentence.

    - Input: a sentence 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": '''You are an expert at named entity recognition. It is your task to identify all the entities in a sentence. These entities can be persons, organizations, locations, events, products, date/time and numbers. For each entity return the sentence with only that entity highlighted. Make sure to place [BoE] before the entity and [EoE] after the entity. Place '\n' between the sentences.'''},
                {"role": "user", "content": '''Sentence: Marseille prosecutor says "so far no videos were used in the crash investigation" despite media reports.'''},
                {"role": "assistant", "content": '''[BoE] Marseille prosecutor [EoE] says "so far no videos were used in the crash investigation" despite media reports.\nMarseille prosecutor says "so far no [BoE] videos [EoE] were used in the crash investigation" despite media reports.\nMarseille prosecutor says "so far no videos were used in [BoE] the crash investigation [EoE]" despite media reports.\nMarseille prosecutor says "so far no videos were used in the crash investigation" despite [BoE] media reports [EoE].'''},
                {"role": "user", "content": '''Sentence: Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June.'''},
                {"role": "assistant", "content": '''[BoE] Membership [EoE] gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June.\nMembership gives the [BoE] ICC [EoE] jurisdiction over alleged crimes committed in Palestinian territories since last June.\nMembership gives the ICC [BoE] jurisdiction [EoE] over alleged crimes committed in Palestinian territories since last June.\nMembership gives the ICC jurisdiction over [BoE] alleged crimes [EoE] committed in Palestinian territories since last June.\nMembership gives the ICC jurisdiction over alleged crimes committed [BoE] in Palestinian territories [EoE] since last June.\nMembership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories [BoE] since last June [EoE].'''},
                {"role": "user", "content": '''Sentence: The number of executions worldwide has gone down by almost 22% compared with 2013, but death sentences up by 28%.'''},
                {"role": "assistant", "content": '''[BoE] The number of executions worldwide [EoE] has gone down by almost 22% compared with 2013, but death sentences up by 28%.\nThe number of executions worldwide has gone [BoE] down by almost 22% compared with 2013 [EoE], but death sentences up by 28%.\nThe number of executions worldwide has gone down by almost 22% compared with 2013, but [BoE] death sentences [EoE] up by 28%.\nThe number of executions worldwide has gone down by almost 22% compared with 2013, but death sentences [BoE] up by 28% [EoE].'''},
                {"role": "user", "content": f'''Sentence: {sentence}'''}
            ]

def create_document_sentences_extractor_messages(document: str, summary: str) -> list:
    '''This function initiates the messages for invoking an LLM response that extracts sentences from a document required to fact-check a statement.
    The conversation list are messages including the system prompt, the document and the summary.
    Here only one example is given because of the limit of the context window.
    
    - Input: a document 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in extracting sentences from source texts. 
                    
                    It is your task to return all sentences from the source text that aid in fact-checking the summary. 
                    If a sentence is necessary to fact-check the statements in the summary, then it should be included in the response.    
                    Copy the sentences entirely and verbatim i.e. word-for-word.
                    Do not copy sentences from the summary.
                    
                    Do not number the sentences. Do not rephrase. 
                    Do not add any additional information or commentary. 
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {document}
                        
                        Summary: {summary}
                        
                        The extracted sentences:
                        '''}
            ]

# Functions for counterfactual debate

def create_hallucination_abduction_messages(knowledge: str, statement: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in detecting hallucinations. 
                    It is your task to explain why a summary contains hallucinations.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                     
                    Given a source text and a summary, it is your task to highlight all the hallucinations.
                    Explain why the statement is hallucinated based on the source text. 
                    Refer to the source text sentences that prove your claim.
                    Make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {knowledge}
                       
                    Summary: {statement}
                    
                    Reasoning:
                    '''}
            ]

def create_supported_abduction_messages(knowledge: str, statement: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in hallucinations. 
                    It is your task to explain why a summary is supported by a source text.
                    This is done by explaining why the summary does not contain hallucinations.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                     
                    Given a source text and a summary, highlight the entailment of each statement.
                    Explain why the statement is supported by the source text. 
                    Refer to the source text sentences that support your claim.
                    Make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {knowledge}
                       
                    Summary: {statement}
                    
                    Reasoning:
                    '''}
            ]

def create_hallucinated_critic_messages(document: str, sentence: str, explanation: str) -> list:
    '''This function initiates the messages for invoking an LLM response that critiques a claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement and the claim.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in hallucinations.
                    It is your task to critique a speaker's argumentation regarding hallucinations in a summary.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    You are given a document, a summary and a speaker's argumentation. 
                    The speaker's argumentation might be wrong and might contain mistakes. 
                    It is your task to persuade the speaker that their argumentation is wrong based on the document.

                    Make sure your persuasion is very short.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}
                       
                       Summary: {sentence}
                       
                       Argumentation: {explanation}
                       
                       Persuasion:
                    '''}
            ]

def create_supported_critic_messages(document: str, sentence: str, explanation: str) -> list:
    '''This function initiates the messages for invoking an LLM response that critiques a claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement and the claim.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are given a document, a summary and a speaker's argumentation. 
                    
                    The speaker's argumentation might be wrong and contain mistakes. 
                    It is your task to persuade the speaker that their argumentation is wrong based on the document.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    The speaker claims that the summary is supported and does not contain hallucinations.
                    Please persuade the speaker that their claim is wrong. 
                    Make sure your persuasion is very short.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}
                       
                       Summary: {sentence}
                       
                       Claim: {explanation}
                       
                       Persuasion:
                    '''}
            ]

def create_defence_hallucination_messages(document: str, sentence: str, explanation: str, critique: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends their claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement, the claim and the critique.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are given a document, a sentence, your past argumentation and someone else their critique on your argumentation. 
                    
                    You believe the summary contains hallucinations based on the document.
                    It is your task to point out the errors in the critique and reiterate your point regarding factual hallucinations. 
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    Please explain the errors in the critique. 
                    Make sure your response is short.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}
                       
                       Summary: {sentence}
                       
                       Your claim: {explanation}
                       
                       Someone's critique: {critique}
                       
                       Your defence:
                    '''}
            ]
  
def create_defence_supported_messages(document: str, sentence: str, explanation: str, critique: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends their claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement, the claim and the critique.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are given a document, a sentence, your past argumentation and someone else their critique on your argumentation. 
                    
                    You believe the summary is supported by the document and does not contain hallucinations.
                    It is your task to point out the errors in the critique and reiterate your point regarding no hallucinations. 
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    Please explain the errors in the critique. 
                    Make sure your response is short.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}
                       
                       Summary: {sentence}
                       
                       Your claim: {explanation}
                       
                       Someone's critique: {critique}
                       
                       Your defence:
                    '''}
            ]  
  
def create_judge_messages(summary: str, debates: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends their claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement, the claim and the critique.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert judge.
                     
                    It is your task to analyze a debate.
                    The debate is about whether or not a summary contains hallucinations. 
                                                            
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    After hearing the arguments of both sides, do you think the summary contains hallucinations or not? 
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If it does not contain hallucinated contents, respond with [SUPPORTED].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''Debates: {debates}

                        Summary: {summary}
                        
                        Judgement:
                    '''}
            ]   

def create_extended_judge_messages(document: str, summary: str, debates: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends their claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement, the claim and the critique.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert judge.
                     
                    It is your task to analyze a debate.
                    The debate is about whether or not a summary of a source text contains hallucinations. 

                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    Base yourself on the source text and the debate.
                    Their argumentation might contain errors.
                    
                    After hearing the arguments of both sides, do you think the summary contains hallucinations or not? 
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If the summary is supported by the document, respond with [SUPPORTED].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}

                        Summary: {summary}
                        
                        Debates: {debates}
                        
                        Judgement about the summary:
                    '''}
            ]   

def counterfactual_debate(debating_LLM: str, document: str, summary: str) -> Tuple[int, str, str]:

    response_function = response_dict[debating_LLM]
    print("-" * 100)
    print("Debates generated with ", debating_LLM)
    # Create the debate claiming hallucinations
    stance_hallucinated = response_function(create_hallucination_abduction_messages(document, summary))
    stance_hallucinated_critique = response_function(create_hallucinated_critic_messages(document, summary, stance_hallucinated))
    stance_hallucinated_defence = response_function(create_defence_hallucination_messages(document, summary, stance_hallucinated, stance_hallucinated_critique))
    debate_hallucinated = f"The debate claiming [HALLUCINATED] :\nClaim: " + stance_hallucinated + "\nCritique: " + stance_hallucinated_critique + "\nDefence: " + stance_hallucinated_defence + "\n" + "End of the debate claiming [HALLUCINATED]."
    print(debate_hallucinated)
    print("-" * 100)
    
    # Create the debate claiming no hallucinations present
    stance_supported = response_function(create_supported_abduction_messages(document, summary))
    stance_supported_critique = response_function(create_supported_critic_messages(document, summary, stance_supported))
    stance_supported_defence = response_function(create_defence_supported_messages(document, summary, stance_supported, stance_supported_critique))
    debate_supported = f"The debate claiming [SUPPORTED] :\nClaim: " + stance_supported + "\nCritique: " + stance_supported_critique + "\nDefence: " + stance_supported_defence + "\n" + "End of the debate claiming [SUPPORTED]."
    print(debate_supported)
    print("-" * 100)
    
    # Append the debates
    debates = "\n" + debate_hallucinated + "\n" + debate_supported
    
    final_judgement = gpt4o_response(create_judge_messages(summary, debates))
    print("The final judgement after counterfactual debating:\n" + final_judgement)

    if "[HALLUCINATED]" in final_judgement:
        return (1, debate_hallucinated, debate_supported)
    
    return (0, debate_hallucinated, debate_supported)
  
def counterfactual_debate_extended(document: str, summary: str, debate: str) -> int:
        
    final_judgement = gpt4o_response(create_extended_judge_messages(document, summary, debate))
    print("The final judgement after counterfactual debating:\n" + final_judgement)

    if "[HALLUCINATED]" in final_judgement:
        return 1
    
    return 0
  
def counterfactual_debate_modified(debating_LLM: str, document: str, summary: str) -> Tuple[int, str, str]:

    response_function = response_dict[debating_LLM]
    print("-" * 100)
    print("Debate generated with ", debating_LLM)
    # Create the debate claiming hallucinations
    stance_hallucinated = response_function(create_hallucination_abduction_messages(document, summary))
    debate_hallucinated = f"The argument in favor of [HALLUCINATED]:\n" + stance_hallucinated + "\n" + "End of the argument in favor of [HALLUCINATED]."
    
    # Create the debate claiming no hallucinations present
    stance_supported = response_function(create_supported_abduction_messages(document, summary))
    debate_supported = f"The argument in favor of [SUPPORTED]:\n" + stance_supported + "\n" + "End of the argument in favor of [SUPPORTED]."
    
    # Append the debates
    debates = "\n" + debate_hallucinated + "\n" + debate_supported + "\n"
    
    print(debates)
    print("-" * 100)

    final_judgement = gpt4o_response(create_extended_judge_messages(document, summary, debates))
    print("The final judgement after counterfactual debating:\n" + final_judgement)

    if "[HALLUCINATED]" in final_judgement:
        return (1, debate_hallucinated, debate_supported)
    
    return (0, debate_hallucinated, debate_supported)  
  
# Functions for collaborative debates  
  
def create_collaboration_messages(knowledge: str, statement: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in detecting hallucinations. 
                    
                    It is your task to identify if a summary contains hallucinations based on the source text.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                     
                    Given a source text and a summary, it is your task to highlight all the statements and analyse if it contains hallucinations.
                    Treat all the statements in the summary.
                    Explain why the statement is hallucinated based on the source text, or why not.
                     
                    Refer to the source text sentences that prove your claim.
                    And make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {knowledge}
                       
                    Summary: {statement}
                    
                    Reasoning:
                    '''}
            ]  

def create_collaboration_feedback_messages(knowledge: str, statement: str, reasoning: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in detecting hallucinations. 
                    
                    You are given another assistant's response to a hallucination detection task.
                    
                    It is your task to identify potential mistakes in their reasoning.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.

                    Did the assistant miss any hallucinations?
                    Or did the assistant wrongfully identify hallucinations?
                    It is your task to provide critical feedback based on the source text.
                    Explain which part of their reasoning is flawed and which part is correct.
                    
                    Refer to the source text sentences to prove your claim.
                    And make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {knowledge}
                       
                    Summary: {statement}
                    
                    Assistant's reasoning: {reasoning}
                    
                    Your feedback:
                    '''}
            ]  
  
def collaborative_debate(debating_LLM: str, document: str, summary: str) -> Tuple[int, str, str]:

    response_function = response_dict[debating_LLM]
    print("-" * 100)
    print("Debate generated with ", debating_LLM)
    # Create the debate claiming hallucinations
    stance_hallucinated = response_function(create_collaboration_messages(document, summary))
    debate_hallucinated = f"The initial analysis of the summary:\n" + stance_hallucinated + "\n" + "End of the initial analysis of the summary."
    
    # Create the debate claiming no hallucinations present
    stance_supported = response_function(create_collaboration_feedback_messages(document, summary, stance_hallucinated))
    debate_supported = f"The feedback on the analysis:\n" + stance_supported + "\n" + "End of the feedback on the analysis."
    
    # Append the debates
    debates = "\n" + debate_hallucinated + "\n" + debate_supported + "\n"
    
    print(debates)
    print("-" * 100)

    final_judgement = gpt4o_response(create_extended_judge_messages(document, summary, debates))
    print("The final judgement after counterfactual debating:\n" + final_judgement)

    if "[HALLUCINATED]" in final_judgement:
        return (1, debate_hallucinated, debate_supported)
    
    return (0, debate_hallucinated, debate_supported) 


# Functions for chain of tailored debates  
  
def create_statement_hallucination_abduction_messages(document: str, summary: str, statement: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in detecting hallucinations. 
                    It is your task to explain why a statement in a summary contains hallucinations.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                     
                    Given a source text, a summary and a statement, it is your task to highlight the hallucinations in the statement.
                    
                    Explain why the statement contains hallucinations based on the source text. 
                    Refer to the source text sentences that support your claim.
                    Make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {document}
                       
                    Summary: {summary}
                    
                    Highlighted statement: [{statement}]
                    
                    Reasoning:
                    '''}
            ]

def create_statement_supported_abduction_messages(document: str, summary: str, statement: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends the claim that the statement contains factual hallucinations.
    The conversation list are messages including the system prompt, 2-shot prompting, the document and the statement.

    - Input: knowledge and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in reasoning.
                     
                    It is your task to explain why a statement in a summary is supported by a source text.
                    This is done by explaining why the highlighted statement does not contain hallucinations.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                     
                    Given a source text, a summary and a statement, explain the entailment of the statement.
                    Explain why the statement is supported by the source text. 
                    Refer to the source text sentences that support your claim.
                    Make sure your explanation is very short and to the point.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {document}
                       
                    Summary: {summary}
                    
                    Highlighted statement: [{statement}]
                    
                    Reasoning: 
                    '''}
            ]  

def create_chain_debates_judge_messages(document: str, summary: str, debate: str) -> list:
    '''This function initiates the messages for invoking an LLM response that defends their claim about factuality hallucinations in a statement.
    The conversation list are messages including the system prompt, the document, the statement, the claim and the critique.

    - Input: a document and a statement 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert judge.
                     
                    It is your task to analyze a source text, a summary, a highlighted statement and a debate about that highlighted statement.
                    The debate is about whether or not that statement is hallucinationed or supported. 
                    One side argues that there are hallucinations in the statement, while the other side argues that the statement is supported by the source text.
                    Base your judgement on the source text and the debate.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                    If the entity can be directly entailed using the information from the source text, then it is non-hallucinated.
                    
                    After hearing both sides of the debate, is the highlighted statement hallucinated or supported?
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If it does not contain hallucinated contents, respond with [SUPPORTED].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}

                        Summary: {summary}
                        
                        Debate: {debate}
                        
                        Judgement:
                    '''}
            ] 

def chain_debates(debating_LLM: str, document: str, summary: str) -> Tuple[int, str, str]:

    sentences = gpt4o_mini_response(create_summary_sentence_extractor_messages(summary)).split("\n")

    debate_history = ""
    
    for statement in sentences:
    
        response_function = response_dict[debating_LLM]
        # Create the stance claiming hallucinations
        stance_hallucinated = response_function(create_statement_hallucination_abduction_messages(document, summary, statement))
        debate_hallucinated = f"The argument claiming [HALLUCINATED] :\nClaim: " + stance_hallucinated + "\nEnd of the debate claiming [HALLUCINATED]."
        
        print("-" * 100)
        print("Debates generated with ", debating_LLM)    
        print(debate_hallucinated)
        print("-" * 100)
        
        # Create the stance claiming no hallucinations present
        stance_supported = response_function(create_statement_supported_abduction_messages(document, summary, statement))
        debate_supported = f"The debate claiming [SUPPORTED] :\nClaim: " + stance_supported + "\nEnd of the debate claiming [SUPPORTED]."
        print(debate_supported)
        print("-" * 100)
        
        # Append the debates
        debate = "\n" + debate_hallucinated + "\n" + debate_supported
        
        debate_history += "The debate about statement" + statement + "\n" + debate + "\n" "- " * 100 + "\n"
        
        final_judgement = gpt4o_response(create_chain_debates_judge_messages(document, summary, debate))
        print("The final judgement after counterfactual debating:\n" + final_judgement)

        if "[HALLUCINATED]" in final_judgement:
            return (1, debate, debate_history)
    
    return (0, debate, debate_history)
  
# Functions for baseline

def create_zeroshot_hallucination_judge(document: str, summary: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying summaries. 
                    
                    It is your task to judge whether a summary contains hallucinated content or not.
                    As soon as a statement in a summary is hallucinated, then the summary contains hallucinated content.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the document. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are statements that contradict the document.
                        
                    If a statement can be directly inferred from the document, then it is supported.
                    If a statement is entailed by the document, then it is also supported.
                    
                    Given a document and a summary, does the summary contain hallucinated content or not?
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If it does not contain hallucinated contents, respond with [SUPPORTED].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''Document: {document}
                       
                       Summary: {summary}
                       
                       Judgement:
                    '''}
            ]   
  
def create_chain_thought_hallucination_judge(document: str, summary: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying summaries. 
                    
                    You are given a document and a summary.
                    It is your task to judge whether the summary contains hallucinated content or not based on the provided document.
                        
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are statements that contradict the document.
                        
                    If the summary can be directly inferred from the document, then it is supported.
                    If the summary is entailed by the document, then it is supported.
                                        
                    Given this document, does the summary contain hallucinated content or not?
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If it does not contain hallucinated contents, respond with [SUPPORTED].
                    
                    First, let's think step-by-step.
                    '''},
                {"role": "user", "content": 
                    f'''[Beginning of document]
                        {document}
                        [End of document]
                       
                       Summary: {summary}
                       
                       Reasoning:
                    '''}
            ]     
  
def baseline(document: str, summary: str) -> int:
    baseline_judgement = gpt4o_mini_response(create_zeroshot_hallucination_judge(document, summary))
    print("The baseline zeroshot judgement:\n" + baseline_judgement)
    if "[HALLUCINATED]" in baseline_judgement:
        return 1
    return 0  

def chain_thoughts(document: str, summary: str) -> Tuple[int, str]:
    thought_judgement = gpt4o_mini_response(create_chain_thought_hallucination_judge(document, summary))
    print("The chain of thought judgement:\n" + thought_judgement)
    if "[HALLUCINATED]" in thought_judgement:
        return (1, thought_judgement)
    return (0, thought_judgement)  

# Functions for knowledge filtering

def create_knowledge_filtered_hallucination_judge(document: str, summary: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying summaries. 
                    
                    It is your task to judge whether a summary contains hallucinated content or not based on the provided source text.
                    As soon as a statement in a summary is hallucinated, then the summary contains hallucinated content.
                    
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the source text nor factual. 
                        Intrinsic hallucinations are statements that contradict the source text.
                        
                    If a statement can be directly inferred from the source text, then it is not hallucinated.
                    If a statement is entailed by the source text, then it is not hallucinated.
                    
                    The source text is filtered to contain only the relevant information for fact-checking the summary.
                    
                    Given this source text and a summary, does the summary contain hallucinated content or not?
                    If it contains hallucinated contents, respond with [TRUE]. 
                    If it does not contain hallucinated contents, respond with [FALSE].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''Source text: {document}
                       
                       Summary: {summary}
                       
                       Judgement:
                    '''}
            ] 

def knowledge_filtering(filtering_LLM: str, document: str, summary: str) -> Tuple[int, str]:
    
    response_function = response_dict[filtering_LLM]
    filtered_document = response_function(create_document_sentences_extractor_messages(document, summary))
    print(f"The filtered document using {filtering_LLM}:\n" + filtered_document)
    
    baseline_judgement = gpt4o_response(create_knowledge_filtered_hallucination_judge(filtered_document, summary))
    print(f"The judgement with filtered knowledge using {filtering_LLM}:\n" + baseline_judgement)
    
    if "[TRUE]" in baseline_judgement:
        return (1, filtered_document)
    return (0, filtered_document)

# Functions for sentence level detection

def create_sentence_level_hallucination_judge(document: str, summary: str, sentence: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying sentences. 
                    
                    You are given a document and a summary with a highlighted sentence.
                    It is your task to judge whether the sentence contains hallucinated content or not based on the provided document.
                    The sentence is part of the summary, however only classify the highlighted sentence. 
                        
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the source text. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are statements that contradict the document.
                        
                    If the sentence can be directly inferred from the document, then it is supported.
                    If the sentence is entailed by the document, then it is supported.
                                        
                    Given this document, does the sentence contain hallucinated content or not?
                    If it contains hallucinated contents, respond with [HALLUCINATED]. 
                    If it does not contain hallucinated contents, respond with [SUPPORTED].
                    
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''[Beginning of document]
                        {document}
                        [End of document]
                       
                       Summary: {summary}
                       
                       Highlighted sentence: [{sentence}]
                       
                       Judgement:
                    '''}
            ] 
    
def sentence_level(judging_LLM: str, document: str, summary: str) -> int:
    response_function = response_dict[judging_LLM]
    sentences = gpt4o_mini_response(create_summary_sentence_extractor_messages(summary)).split('\n')
    
    for highlighted_sentence in sentences:
        print("-" * 25)
        #print(f"The highlighted sentence using {judging_LLM}:\n" + highlighted_sentence)
        print(f"The highlighted sentence:\n" + highlighted_sentence)

        partial_judgement = response_function(create_sentence_level_hallucination_judge(document, summary, highlighted_sentence))
        print(f"The partial judgement with sentence level detection using {judging_LLM}:\n" + partial_judgement)
        
        if "HALLUCINATED" in partial_judgement:
            return 1
    return 0

# Functions for statement level detection

def create_statement_level_hallucination_judge(document: str, summary: str, highlighted_sentence: str, isolated_statement: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying statements in either hallucinated and supported.
                    
                    You are given a document.
                    You are also given a summary of the document, a highlighted sentence from the summary and an isolated statement from that summary's sentence.
                    It is your task to judge whether the isolated statement is hallucinated or supported, based on the document.
                        
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the document. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are statements that contradict the document.
                    
                    On the other hand, if the isolated statement can be inferred from the document in its entirety, then it is supported.
                    Or if the isolated statement is directly entailed by the document, then it is supported.
                                        
                    Given the document, does the isolated statement contain hallucinations or is it supported?
                    
                    Respond with [HALLUCINATED] or with [SUPPORTED].
                    Do not give an explanation.
                    '''},
                {"role": "user", "content": 
                    f'''[Beginning of document]
                        {document}
                        [End of document]
                        
                        Summary: {summary}
                        
                        Highlighted sentence: "{highlighted_sentence}"
                        
                        Isolated statement: "{isolated_statement}"
                       
                       Judgement of the statement:
                    '''}
            ] 

def statement_level(judging_LLM: str, document: str, summary: str) -> int:
    
    response_function = response_dict[judging_LLM]
    sentences = gpt4o_mini_response(create_summary_sentence_extractor_messages(summary)).split('\n')
    
    for highlighted_sentence in sentences:
        print("-" * 25)
        print(f"The highlighted sentence:\n" + highlighted_sentence)
        
        statements = gpt4o_mini_response(create_sentence_statement_extractor_messages(summary, highlighted_sentence)).split('\n')
        
        for highlighted_statement in statements: 
            print("-" * 25)
            #print(f"The highlighted sentence using {judging_LLM}:\n" + highlighted_sentence)
            print(f"The highlighted statement:\n" + highlighted_statement)

            partial_judgement = response_function(create_statement_level_hallucination_judge(document, summary, highlighted_sentence, highlighted_statement))
            print(f"The partial judgement with statement level detection using {judging_LLM}:\n" + partial_judgement)
            
            if "HALLUCINATED" in partial_judgement:
                return 1
    return 0

# Functions for chain of tailored thoughts

def create_chain_tailored_thoughts_hallucination_judge(document: str, summary: str, highlighted_sentence: str, isolated_statement: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying statements in either hallucinated and supported.
                    
                    You are given a document.
                    You are also given a summary of the document, a highlighted sentence from the summary and an isolated statement from that summary's sentence.
                    It is your task to judge whether the isolated statement is hallucinated or supported, based on the document.
                        
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the document. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are statements that contradict the document.
                    
                    On the other hand, if the isolated statement can be inferred from the document in its entirety, then it is supported.
                    Or if the isolated statement is directly entailed by the document, then it is supported.
                                        
                    Given the document, does the isolated statement contain hallucinations or is it supported?
                    
                    Respond with [HALLUCINATED] or with [SUPPORTED].
                    
                    First, let's think step-by-step.
                    '''},
                {"role": "user", "content": 
                    f'''[Beginning of document] 
                        {document}
                        [End of document]
                        
                        Summary: {summary}
                        
                        Highlighted sentence: {highlighted_sentence}
                        
                        Isolated statement: "{isolated_statement}"
                       
                        Reasoning:
                    '''}
            ] 

def chain_tailored_thoughts(judging_LLM: str, document: str, summary: str) -> Tuple[int, str]:
    
    response_function = response_dict[judging_LLM]
    sentences = gpt4o_mini_response(create_summary_sentence_extractor_messages(summary)).split('\n')
    
    for highlighted_sentence in sentences:
        print("-" * 25)
        print(f"The highlighted sentence:\n" + highlighted_sentence)
        
        statements = gpt4o_mini_response(create_sentence_statement_extractor_messages(summary, highlighted_sentence)).split('\n')
        
        for highlighted_statement in statements: 
            print("-" * 25)
            #print(f"The highlighted sentence using {judging_LLM}:\n" + highlighted_sentence)
            print(f"The highlighted statement:\n" + highlighted_statement)

            partial_judgement = response_function(create_chain_tailored_thoughts_hallucination_judge(document, summary, highlighted_sentence, highlighted_statement))
            print(f"The partial judgement with statement level detection using {judging_LLM}:\n" + partial_judgement)
            
            if "HALLUCINATED" in partial_judgement:
                return (1, partial_judgement)
    return (0, partial_judgement)

def create_chain_tailored_thoughts_sentence_hallucination_judge(document: str, summary: str, highlighted_sentence: str) -> list:
    '''This function initiates the messages for invoking the baseline LLM response when judging whether a summary is hallucinated or not.
    The conversation list are messages including the system prompt, the document and the summary.

    - Input: a document and a summary 
    - Output: the conversation list 
    '''
    return [
                {"role": "system", "content": 
                    '''You are an expert in classifying sentences in either hallucinated and supported.
                    
                    You are given a document.
                    You are also given a summary of the document and a highlighted sentence from the summary.
                    It is your task to judge whether the highlighted sentence is hallucinated or supported, based on the document.
                        
                    There are three types of hallucinations; 
                        Factual hallucinations refer to content that might be verifiable by world knowledge but is not inferable from the document. 
                        Non-factual hallucinations are entities that are neither inferable from the document nor factual. 
                        Intrinsic hallucinations are sentences that contradict the document.
                    
                    On the other hand, if the highlighted sentence can be inferred from the document in its entirety, then it is supported.
                    Or if the highlighted sentence is directly entailed by the document, then it is supported.
                                        
                    Given the document, does the highlighted sentence contain hallucinations or is it supported?
                    
                    Respond with [HALLUCINATED] or with [SUPPORTED].
                    
                    First, let's think step-by-step.
                    '''},
                {"role": "user", "content": 
                    f'''[Beginning of document] 
                        {document}
                        [End of document]
                        
                        Summary: {summary}
                        
                        Highlighted sentence: "{highlighted_sentence}"
                                               
                        Reasoning:
                    '''}
            ] 
    
def chain_tailored_thoughts_sentence(judging_LLM: str, document: str, summary: str) -> Tuple[int, str]:
    
    response_function = response_dict[judging_LLM]
    sentences = gpt4o_mini_response(create_summary_sentence_extractor_messages(summary)).split('\n')
    
    for highlighted_sentence in sentences:
        print("-" * 25)
        print(f"The highlighted sentence:\n" + highlighted_sentence)
        
        partial_judgement = response_function(create_chain_tailored_thoughts_sentence_hallucination_judge(document, summary, highlighted_sentence))
        print(f"The partial judgement with statement level detection using {judging_LLM}:\n" + partial_judgement)
        
        if "HALLUCINATED" in partial_judgement:
            return (1, partial_judgement)
    return (0, partial_judgement)

# Extracting SummEval summaries

def find_random_summary_with_consistency_5(row):
    consistency_scores = row['consistency']
    summaries = row['machine_summaries']
    summaries_with_score_5 = [summaries[idx] for idx, score in enumerate(consistency_scores) if score == 5]
    if summaries_with_score_5:
        return random.choice(summaries_with_score_5)
    return None

def find_random_summary_with_hallucinations(row):
    consistency_scores = row['consistency']
    summaries = row['machine_summaries']
    summaries_with_hallucination = [summaries[idx] for idx, score in enumerate(consistency_scores) if score < 5]
    if summaries_with_hallucination:
        return random.choice(summaries_with_hallucination)
    return None