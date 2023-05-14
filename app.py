import numpy as np
import streamlit as st
from get_videos import get_videos
import json
from get_transcript import get_transcript_summary, raw_transcript_time_estimator
from streamlit.components import v1 as components
from qa_system import answer_the_question, generate_question_answer, estimate_question_toughness, best_question, compare_answers
import os
import pickle as pkl
import gzip
import hashlib
from glob import glob

CACHE_DIR = './.cache/'

os.makedirs(CACHE_DIR, exist_ok=True)


selected_page = None
transcript=None

def sha256(string):
    hash_obj = hashlib.sha256(string.encode())
    hex_dig = hash_obj.hexdigest()
    return hex_dig
#     bytes_obj = bytes.fromhex(hex_dig)
#     str_obj = bytes_obj.decode('iso-8859-1')
#     return str_obj

def check_cache_existance(string):
    existing_files = glob("{}/*".format(CACHE_DIR))
    hash_it = sha256(string)
    
#     st.write(hash_it)
#     st.write(existing_files)
    
    for h in existing_files:
        if hash_it in h: 
            return True, hash_it
    return False, hash_it

def write_pickle_compressed(filename, data):
    with gzip.open(filename, 'wb') as f:
        pkl.dump(data, f)

def load_pickle_compressed(filename):
    with gzip.open(filename, 'rb') as f:
        data = pkl.load(f)
    return data

def intro():
  d = {}
  st.title('Learn from AI')
  playlist_name = st.text_input(
    'Enter the playlist link you want to learn from today')
  if playlist_name:
    all_videos = get_videos(playlist_name)
    urls = all_videos["url"]
    titles = all_videos["title"]
    #st.json(all_videos)
    #json.dump(all_videos, open("videos.txt",'w'))
    global selected_page
    selected_page = st.sidebar.selectbox("Index", titles)
    for i in range(len(urls)):
      url = all_videos["url"][i]
      title = all_videos["title"][i]
      d[title] = url
    json.dump(d, open("{}/videos.txt".format(CACHE_DIR), 'w'))

def seconds_to_hms(seconds):
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    seconds = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def data_frame_demo(d, selected_page):
    st.video(d[selected_page])
    
    global transcript
    
    flag, hash_it = check_cache_existance(d[selected_page])
    if flag:
        data = load_pickle_compressed(CACHE_DIR + hash_it)
        transcript, result, summary = data['transcript'], data['result'], data['summary']
        
    else:
        transcript, result, summary = get_transcript_summary(d[selected_page])
        data = {'transcript':transcript, 'result':result, 'summary':summary}
        write_pickle_compressed(CACHE_DIR + hash_it, data)
        
    #st.write(transcript)
    st.write(summary)
    question = st.text_input('Do you have any question?')
    if question:
        flag, hash_it = check_cache_existance(d[selected_page] + ' ' + question)
        if flag:
            data = load_pickle_compressed(CACHE_DIR + hash_it)
            ans = data['ans']

        else:
            ans = answer_the_question(question, result)
            data = {'ans':ans}
            write_pickle_compressed(CACHE_DIR + hash_it, data)

        
        st.write(ans[0])
        time = raw_transcript_time_estimator(int(ans[1][0]),int(ans[1][1]),transcript)
        st.write('You can also refer the video at this timestamp:',seconds_to_hms(time[0]),'to',seconds_to_hms(time[1]))
    if st.checkbox("Do you want to practice?"):
        flag, hash_it = check_cache_existance(d[selected_page] + ' ' + 'RAW DATA')
        if flag:
            data = load_pickle_compressed(CACHE_DIR + hash_it)
            qa_list, css = data['qa_list'], data['css']

        else:
            qa_list = generate_question_answer(result)
            css = estimate_question_toughness(qa_list, result)
            data = {'qa_list':qa_list, 'css':css}
            write_pickle_compressed(CACHE_DIR + hash_it, data)
        
        
        difficulty_level = st.text_input('On a scale 1-10 what level of questions do you want?')
        if difficulty_level:
            #print(qa_list)
            #print(css)
            best = best_question(css,int(difficulty_level))
            st.write(qa_list[best]['question'])
            user_ans = st.text_input('Give you answer')
            if user_ans:
                if not compare_answers(user_ans,qa_list[best]['answer']):
                    st.write('Sorry wrong answer, the correct answer is:',qa_list[best]['answer'])
                else:
                    st.write('Correct answer, Congrats!!')
            
            

intro()
if selected_page:
    d = json.load(open("{}/videos.txt".format(CACHE_DIR)))
    data_frame_demo(d, selected_page)