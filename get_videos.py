from pytube import YouTube, Playlist

def get_videos(URL_PLAYLIST):
    playlist = Playlist(URL_PLAYLIST)
    info = {'url':[], 'title':[]}
    for url in playlist:
        yt = YouTube(url)
        info['url'].append(url)
        info['title'].append(yt.title)
    return info


if __name__ == '__main__':
    URL_PLAYLIST = 'https://www.youtube.com/playlist?list=PLp6ek2hDcoNB_YJCruBFjhF79f5ZHyBuz'
    print(get_videos(URL_PLAYLIST))