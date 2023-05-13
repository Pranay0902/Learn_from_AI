from transformers import pipeline

def answer_question(question, context, model='question-answering'):
    qa_model = pipeline(model)
    return qa_model(question = question, context = context)

if __name__ == '__main__':
    question = "What is tallest tower in paris?"
    context = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    print(answer_question(question, context))
