from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from pprint import pprint
from transformers import pipeline
import torch
import random

def get_transcript_summary(youtube_link, hf_name = 'pszemraj/led-large-book-summary', MAX_LEN=16384):
    json_formatter = JSONFormatter()

    video_id = youtube_link.split("=")[1]

    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
#     json_transcript = json_formatter.format_transcript(transcript=raw_transcript)
    
    pprint(raw_transcript)
    
    result = ""
    for i in raw_transcript:
        result += ' ' + i['text']

    summarizer = pipeline(
        "summarization",
        hf_name,
        device= random.randint(0, 3) if torch.cuda.is_available() else -1,
    )

    if len(result) > MAX_LEN:
        result = result[:MAX_LEN]
    
    summarized_text = summarizer(
        result,
        min_length=16,
        max_length=1000,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )[0]['summary_text']
    
    del summarizer
    
    return raw_transcript, result, summarized_text

def binary_search(arr, x):
    """
    Returns the index of x in arr if present, else returns an estimate of its index.
    """
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    # x is not present in arr, so we return an estimate of its index
    return left

def raw_transcript_time_estimator(start_idx, end_idx, raw_transcript):
    info_dict = {'start_idx':[], 'end_idx':[], 'start_time':[], 'end_time':[]}
    
    i = 0
    for e in raw_transcript:
        info_dict['start_idx'].append(i)
        i = i + len(e['text'])
        
        info_dict['start_time'].append(e['start'])
        info_dict['end_time'].append(e['start'] + e['duration'])
        
#     print(info_dict)
    
#     i = 0
#     j = 0
    
#     for idx, s in enumerate(info_dict['start_idx']):
#         if start_idx > s:
#             continue
#         elif start_idx == s:
#             i = idx
#         else:
#             i = idx - 1 if idx > 0 else 0
    
#     for idx, e in enumerate(info_dict['end_idx'][i:]):
#         if end_idx > e:
#             continue
#         else:
#             j = idx + i
    i, j = binary_search(info_dict['start_idx'], start_idx), binary_search(info_dict['start_idx'][1:] + [len(raw_transcript[-1]['text'])], end_idx)
    
#     print(i, j)
    
    return info_dict['start_time'][i], info_dict['end_time'][j]

if __name__ == '__main__':
    URL = 'https://www.youtube.com/watch?v=dRIhrn8cc9w'
#     URL = 'https://www.youtube.com/watch?v=C7OQHIpDlvA'
    
    raw_transcript, context, summarized_text = get_transcript_summary(URL)
    
    pprint(context)
    
    from qa_system import *

    
    qa_list = generate_question_answer(context)
    print(qa_list)
    css = estimate_question_toughness(qa_list, context)
    print(css)
    j = best_question(css, 5)
    print(j)
    
    question = qa_list[j]['question']
    
    answer, pos = answer_the_question(question, context)
    
    print("Question: ", question, "\nAnswer: ", answer, "Position: ", pos)
    st, et = raw_transcript_time_estimator(pos[0], pos[1], raw_transcript)
    print("Position in video: {} to {}".format(st, et))