from transformers import pipeline
from pipelines import pipeline

def answer_the_question(question, context, task="multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl"):
    qa_model = pipeline(task, model=model)
    return qa_model({'question':question, 'context':context})

def generate_question_answer(context, task="question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend"):
    qa_model = pipeline(task, model=model, qg_format=qg_format)
    d = qa_model(context)
    for i in range(len(d)):
        d[i]['answer'] = d[i]['answer'].split('<pad> ')[1]
    return d
    

if __name__ == '__main__':
    question = "What is tallest tower in paris?"
    context = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    print(answer_the_question(question, context))
    print(generate_question_answer(context))
