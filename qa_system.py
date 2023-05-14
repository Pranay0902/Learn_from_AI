from transformers import pipeline
from pipelines import pipeline as pipeline_p
from sentence_transformers import SentenceTransformer, util
import numpy as np
import openai

OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY
max_tokens = 10
TOTAL_MAX = 2000

def ask_chagpt(prompt, max_tokens=max_tokens):
    if len(prompt) > TOTAL_MAX:
        prompt = prompt[:TOTAL_MAX]
        
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    answer = response.choices[0].text.strip()
    return answer

def answer_the_question(question, context, task="question-answering", model="distilbert-base-uncased-distilled-squad",
                       chatgpt=True):
    if chatgpt:
        return ask_chagpt("{}?".format(question), max_tokens=100), (1, 4)
        
    qa_model = pipeline(task, model=model)
    preds = qa_model({'question':question, 'context':context})
    
    del qa_model
    return preds['answer'], (preds['start'], preds['end'])

def generate_question_answer(context, max_lim=4, task="question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend", chatgpt=False):
    
    if chatgpt:
        return ask_chagpt("Generate {} pairs of Question and Answers as a list of dictionary, where each element has 'question' and 'answer' keys on the following context {}.\n ".format(max_lim, context), max_tokens=100*max_lim)
    
    qa_model = pipeline_p(task, model=model, qg_format=qg_format)
    d = qa_model(context)
    for i in range(len(d)):
        d[i]['answer'] = d[i]['answer'].split('<pad> ')[1]
    
    del qa_model
    if len(d) > max_lim:
        d = d[:max_lim]
    return d

def estimate_question_toughness(qa_list, context, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    
    model = SentenceTransformer(model_name)
        
    embedding_1 = model.encode([e['answer'] for e in qa_list], convert_to_tensor=True)
    embedding_2 = model.encode([answer_the_question(e['question'], e['answer']) for e in qa_list], convert_to_tensor=True)
    
    cos_scores = np.diag(util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy())
    
    del model
    return cos_scores

def compare_answers(actual_ans, given_ans, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    
    model = SentenceTransformer(model_name)
        
    embedding_1 = model.encode([actual_ans], convert_to_tensor=True)
    embedding_2 = model.encode([given_ans], convert_to_tensor=True)
    
    cos_scores = np.diag(util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy())
    
    del model

    if cos_scores[0] > 0.85:
        return True
    return False


def best_question(cos_scores, difficulty):
    # difficulty : 0 - 10
    
    diff_norm = difficulty/5 - 1
    e = np.abs(diff_norm - cos_scores)
    return e.argmin()
                    

    
if __name__ == '__main__':
    question = "What is tallest tower in paris?"
    context = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
#     print(answer_the_question(question, context))
#     qa_list = generate_question_answer(context)
#     print(qa_list)
#     css = estimate_question_toughness(qa_list, context)
#     print(css)
#     print(best_question(css, 9))
    
#     print(ask_chagpt("What is difference between RL and ML?", 100))
    
#     print(answer_the_question(question, context))
    qa_list = generate_question_answer(context)
    print(qa_list)
    