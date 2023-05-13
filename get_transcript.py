from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from pprint import pprint
from transformers import pipeline
import torch

def get_transcript_summary(youtube_link, hf_name = 'pszemraj/led-large-book-summary'):
    json_formatter = JSONFormatter()

    video_id = youtube_link.split("=")[1]

    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    json_transcript = json_formatter.format_transcript(transcript=raw_transcript)

    result = ""
    for i in raw_transcript:
        result += ' ' + i['text']

    summarizer = pipeline(
        "summarization",
        hf_name,
        device=3 if torch.cuda.is_available() else -1,
    )


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

    return json_transcript, str(summarized_text)

if __name__ == '__main__':
    URL = 'https://www.youtube.com/watch?v=hPIzgZ16oac'
    
    pprint(get_transcript_summary(URL))