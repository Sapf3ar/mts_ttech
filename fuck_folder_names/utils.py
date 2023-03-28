from subprocess import check_call
from mkv import merge, source
import os
import pathlib
import sys
import re
import json
import subprocess


from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchaudio
import num2words


import json
import requests
from IPython.display import clear_output

from pydub import AudioSegment


import torch
from omegaconf import OmegaConf
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
OmegaConf.load('latest_silero_models.yml');


class myMKV(merge.MkvMerge):

    def create(self):
        check_call([self.command] + list(map(str, self.arguments)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                    model='silero_tts',
                                    language='ru',
                                    speaker='v3_1_ru')
audio_model.to(device)


def to_ssml(text):
    start = '''
              <speak>
              <p>
            '''

    end = '''
            </p>
            </speak>
          '''

    return start + text + end


def num_to_words(text):
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index], lang='ru')
    numbers_to_words = ' '.join(after_spliting)
    return numbers_to_words


def text2audio(texts, model, speaker='xenia', sample_rate=48000, ssml=False, put_accent=True, put_yo=True, save=False,
               output_path='out_audio.wav', format='wav', bits_per_sample=64, encoding="PCM_S"):
    '''
    speakers   :  'aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'
    sample_rate:  8000, 24000, 48000
    '''

    for i in range(len(texts)):
        texts[i] = num_to_words(texts[i])
        texts[i] = to_ssml(texts[i])

        if ssml:
            audio = model.apply_tts(ssml_text=texts[i],
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    put_accent=put_accent,
                                    put_yo=put_yo)
        else:
            audio = model.apply_tts(text=texts[i],
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    put_accent=put_accent,
                                    put_yo=put_yo)

        if save:
            torchaudio.save(filepath=f'{output_path}/audio_{i}.wav',
                            format=format,
                            src=audio[None, :],
                            sample_rate=sample_rate,
                            encoding=encoding,
                            bits_per_sample=bits_per_sample)

    # return audio[None,:]



def get_audio(input_path, output_path='out_audio.wav'):
    audioclip = AudioFileClip(input_path)
    audioclip.write_audiofile(codec='pcm_s32le', filename=output_path)


def audio_merge(main_file, second_path, timings=[0], output_file='output.wav', format='wav'):

    main_sound = AudioSegment.from_file(main_file, format=format)

    for i, file in enumerate(second_path):
        main_sound = main_sound.overlay(AudioSegment.from_file(file, format=format), position=timings[i] * 1000)

    file_handle = main_sound.export("output.wav", format=format)

def add_subs_to_file(path_file, path_subs, output_path):

    v_source = get_source(path_file)

    path, ext = path_file.split('.')
    pathsub, extsub = path_file.split('.')

    merge_file: merge.MkvMerge = create_merge_obj(output_path, output_path)

    merge_file.add_source(v_source)
    merge_file.add_subtitle(path_subs,
                            name='my subtitles',
                            is_forced=False,
                            is_default=False,
                            language_code='ru')

    merge_file.create()


def get_source(ep_file):
    """Returns a MkvSource object based on the given input"""
    v_source = source.MkvSource(os.path.join(ep_file))
    v_source.copy_audios('all')
    v_source.copy_videos('all')
    v_source.copy_subtitles('all')
    return v_source


def create_merge_obj(output_folder, ep_file):
    """Returns a MkvMerge object based on the given input"""

    merge_file = myMKV(os.path.join(output_folder, ep_file))

    return merge_file

def add_audio_to_file(input_file, audio_file, output_file):

    subprocess.call(['ffmpeg', '-i', input_file, '-i', input_file, '-i', audio_file, '-c', 'copy', '-map', '0:v', '-map', '1:a', '-map', '3:a', output_file])



def get_mkv_track_id(file_path):
    """ Returns the track ID of the SRT subtitles track"""
    try:
        raw_info = subprocess.check_output(["mkvmerge", "-i", file_path],
                                            stderr=subprocess.STDOUT).decode("utf-8")
    except subprocess.CalledProcessError as ex:
        print(ex)
        sys.exit(1)
    #pattern = re.compile('.* (\d+): subtitles \(SubRip/SRT\).*', re.DOTALL)
    #m = re.findall(r'.* (\d+): subtitles \(SubRip/SRT\)', str(raw_info))

    raws = raw_info.split('\n')

    arr_of_id = []
    for raw in raws:
        if 'subtitles (SubRip/SRT)' in raw:
            try:
                arr_of_id.append(int(raw[9]))
            except:
                print('now way bro')

    return arr_of_id


def extract_mkv_subs(file):
    output_path = file['srt_full_path']
    for i, track_id in enumerate(file['srt_track_ids']):
        subprocess.call(["mkvextract", "tracks", file['full_path'],
                            str(track_id) + ":" + f'{output_path}_subtitles_{i+1}.srt'])



def extract_subs(files):
    for file in files:
        extract_mkv_subs(file)


def get_srt(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
            (basename, ext) = os.path.splitext(name)
            track_ids = get_mkv_track_id(os.path.join(root, name))
        srt_full_path = os.path.join(root, basename)
        file_list.append({
                        'full_path': os.path.join(root, name),
                        'srt_track_ids': track_ids,
                        'srt_full_path': srt_full_path,
                        })
    extract_subs(file_list)


def my_timings(subtitles_path):
    # read file line by line
    file = open(subtitles_path, "r")
    lines = file.readlines()
    file.close()

    timings = []

    for line in lines:
        if '-->' in line:
            timings.append([line[:11], line[17:-2]])

    return timings


def get_timings(srt_path, free=True, in_seconds=True):
    '''
    free = True  -->  return timings when there is no dialogue  (do or do not there is no dialogue)
    free = True  -->  return dialog timings

    in_seconds   -->  return timings like [123, 234] (in seconds from start)
    '''

    dialogue_timings = my_timings(srt_path)

    if free:

        free_timings = []
        is_start = True

        for timing in dialogue_timings:

            hour_start = int(timing[0][1])
            hour_end = int(timing[1][1])

            minute_start_1 = int(timing[0][3])
            minute_start_2 = int(timing[0][4])
            minute_end_1 = int(timing[1][3])
            minute_end_2 = int(timing[1][4])

            second_start_1 = int(timing[0][6])
            second_start_2 = int(timing[0][7])
            second_end_1 = int(timing[1][6])
            second_end_2 = int(timing[1][7])

            sub_second_start_1 = int(timing[0][9])
            sub_second_start_2 = int(timing[0][10])
            sub_second_end_1 = int(timing[1][9])
            sub_second_end_2 = int(timing[1][10])

            if in_seconds:

                time_start = hour_start * 60 * 60 + minute_start_1 * 10 * 60 + minute_start_2 * 60 + second_start_1 * 10 + second_start_2 + (
                            sub_second_start_1 * 10 + sub_second_start_2) * 0.01
                time_end = hour_end * 60 * 60 + minute_end_1 * 10 * 60 + minute_end_2 * 60 + second_end_1 * 10 + second_end_2 + (
                            sub_second_end_1 * 10 + sub_second_end_2) * 0.01

                if is_start:
                    previous_time = 0.0
                    is_start = False

                free_timings.append([float(previous_time), float
                (time_start)])

                previous_time = time_end

            else:
                if is_start:
                    free_timings.append([f'0{0}:{0}{0}:{0}{0},{0}{0}',
                                         f'0{hour_start}:{minute_start_1}{minute_start_2}:{second_start_1}{second_start_2},{sub_second_start_1}{sub_second_start_2}'])
                    is_start = False
                else:
                    free_timings.append([
                                            f'0{previous_hour}:{previous_minute_1}{previous_minute_2}:{previous_second_1}{previous_second_2},{previous_sub_second_1}{previous_sub_second_2}',
                                            f'0{hour_start}:{minute_start_1}{minute_start_2}:{second_start_1}{second_start_2},{sub_second_start_1}{sub_second_start_2}'])

                previous_hour = hour_end

                previous_minute_1 = minute_end_1
                previous_minute_2 = minute_end_2

                previous_second_1 = second_end_1
                previous_second_2 = second_end_2

                previous_sub_second_1 = sub_second_end_1
                previous_sub_second_2 = sub_second_end_2

        return free_timings

    else:
        if in_seconds:
            timings = []

            for timing in dialogue_timings:
                hour_start = int(timing[0][1])
                hour_end = int(timing[1][1])

                minute_start_1 = int(timing[0][3])
                minute_start_2 = int(timing[0][4])
                minute_end_1 = int(timing[1][3])
                minute_end_2 = int(timing[1][4])

                second_start_1 = int(timing[0][6])
                second_start_2 = int(timing[0][7])
                second_end_1 = int(timing[1][6])
                second_end_2 = int(timing[1][7])

                sub_second_start_1 = int(timing[0][9])
                sub_second_start_2 = int(timing[0][10])
                sub_second_end_1 = int(timing[1][9])
                sub_second_end_2 = int(timing[1][10])

                time_start = hour_start * 60 * 60 + minute_start_1 * 10 * 60 + minute_start_2 * 60 + second_start_1 * 10 + second_start_2 + (
                            sub_second_start_1 * 10 + sub_second_start_2) * 0.01
                time_end = hour_end * 60 * 60 + minute_end_1 * 10 * 60 + minute_end_2 * 60 + second_end_1 * 10 + second_end_2 + (
                            sub_second_end_1 * 10 + sub_second_end_2) * 0.01

                timings.append([float(time_start), float(time_end)])
            return timings

        else:
            return dialogue_timings


def cut_by_timings(path, timings, output_folder_path):
    clip = VideoFileClip(path)

    part = 1
    for timing in timings:
        t_start = timing[0]
        t_end = timing[1]
        if t_end - t_start > 3:
            part += 1
            cut = clip.subclip(t_start, t_end)
            cut.write_videofile(f"{output_folder_path}/part_{part}.mp4", codec='libx264', audio=False, verbose=False)