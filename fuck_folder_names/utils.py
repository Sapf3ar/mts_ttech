from subprocess import check_call
from mkv import merge, source
import os
import pathlib
import sys
import re
import json
import subprocess
import glob
import cv2
from torchvision import transforms
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
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from typing import List, Any


class myMKV(merge.MkvMerge):

    def create(self):
        check_call([self.command] + list(map(str, self.arguments)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_audio(input_path, output_path='out_audio.wav'):
    audioclip = AudioFileClip(input_path)
    audioclip.write_audiofile(codec='pcm_s32le', filename=output_path)


def audio_merge(main_file, second_path, timings, output_file='output.wav', format='wav'):
    main_sound = AudioSegment.from_file(main_file, format=format)

    for i, file in enumerate(second_path):
        # timing
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
    subprocess.call(
        ['ffmpeg', '-i', input_file, '-i', input_file, '-i', audio_file, '-c', 'copy', '-map', '0:v', '-map', '1:a',
         '-map', '3:a', output_file])


def get_mkv_track_id(file_path):
    """ Returns the track ID of the SRT subtitles track"""
    try:
        raw_info = subprocess.check_output(["mkvmerge", "-i", file_path],
                                           stderr=subprocess.STDOUT).decode("utf-8")
    except subprocess.CalledProcessError as ex:
        print(ex)
        sys.exit(1)
    # pattern = re.compile('.* (\d+): subtitles \(SubRip/SRT\).*', re.DOTALL)
    # m = re.findall(r'.* (\d+): subtitles \(SubRip/SRT\)', str(raw_info))

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
                         str(track_id) + ":" + f'{output_path}_subtitles_{i + 1}.srt'])


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


def get_timings_for_sum(timecodes):
    fold_timings = []
    for timing in timecodes:
        minute_start_1 = int(str(timing[0])[3])
        minute_start_2 = int(str(timing[0])[4])
        minute_end_1 = int(str(timing[1])[3])
        minute_end_2 = int(str(timing[1])[4])

        second_start_1 = int(str(timing[0])[6])
        second_start_2 = int(str(timing[0])[7])
        second_end_1 = int(str(timing[1])[6])
        second_end_2 = int(str(timing[1])[7])

        sub_second_start_1 = int(str(timing[0])[9])
        sub_second_start_2 = int(str(timing[0])[10])
        sub_second_start_3 = int(str(timing[0])[11])
        sub_second_end_1 = int(str(timing[1])[9])
        sub_second_end_2 = int(str(timing[1])[10])
        sub_second_end_3 = int(str(timing[1])[11])

        time_start = minute_start_1 * 10 * 60 + minute_start_2 * 60 + second_start_1 * 10 + second_start_2 + (
                    sub_second_start_1 * 10 + sub_second_start_2 + sub_second_start_3 * 0.1) * 0.01
        time_end = minute_end_1 * 10 * 60 + minute_end_2 * 60 + second_end_1 * 10 + second_end_2 + (
                    sub_second_end_1 * 10 + sub_second_end_2 + sub_second_end_3 * 0.1) * 0.01
        time = time_end - time_start
        fold_timings.append(time)
    return fold_timings


def cut_by_timings(path, timings, output_folder_path):
    clip = VideoFileClip(path)
    new_timings = []
    part = 1
    for timing in timings:
        t_start = timing[0]
        t_end = timing[1]
        if t_end - t_start > 3:
            new_timings.append([t_start, t_end])
            part += 1
            cut = clip.subclip(t_start, t_end)
            cut.write_videofile(os.path.join(output_folder_path,
                                             f"part_{str(t_start).split('.')[0]}_{str(t_start).split('.')[1]}_{str(t_end).split('.')[0]}_{str(t_end).split('.')[1]}.mp4"),
                                codec='libx264', audio=False, verbose=False)
    return new_timings


def split_video_into_scenes(video_path, save_path, threshold=30.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.auto_downscale = False
    scene_manager.downscale = 1
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    pruned_scenes = []
    for start, end in scene_list:
        if int(end) - int(start) < 4:
            pass
        else:
            pruned_scenes.append([start, end])
    os.chdir(save_path)
    split_video_ffmpeg(video_path, pruned_scenes, show_progress=False)
    return pruned_scenes


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def make_dir(dir_path):
    try:

        os.mkdir(dir_path)

    except OSError:

        pass

    return dir_path


def global_scene_cut(path_to_cut_videos: str):
    video_folders = []
    timecodes = []
    for file_path in glob.glob(os.path.join(path_to_cut_videos, '*.mp4')):
        make_dir(file_path[:-4])
        os.chdir(file_path[:-4])
        timecodes.append(split_video_into_scenes(file_path, save_path=file_path[:-4]))
        os.chdir("..")
        video_folders.append(file_path[:-4])
    return timecodes, video_folders


def question_set_inf(model, frames):
    questions = [

        'What is the main event on a video?',
        "What is shown on the picture?",
        "What humans are doing on a video?",
        "Where are the main objects on a video are located?",
        "What actions are perfomed on a video?",
        'Is there any humans on the picture? Where are they located?',
        "What are the main objects on a video?"
        "How does scene changes throughout the video?"
        'How much humans are on the photo?',
        "How much non-human objects are on the photo?"
        "What are the main non-human objects are on the photo?",
        'How much people are on the photo? Answer with one number'

    ]
    texts = []
    texts.append(model.caption(frames, min_len=15, max_len=200))
    for q in questions:
        texts.append(model.answer(frames, q, min_len=10, max_len=200))
    return texts


def read_video(path, transform=None, frames_num=1):
    frames = []
    cap = cv2.VideoCapture(path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"{length=} {fps=}")
    N = length // (frames_num)
    # N=5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    current_frame = 1
    for i in range(length):
        ret, frame = cap.read(current_frame)

        if ret and i == current_frame and len(frames) < frames_num:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_CUBIC)
            frame = transform(frame).unsqueeze(0).to(device)
            frames.append(frame)
            current_frame += N

    cap.release()
    return frames


def blip_scene_inf(model, folder, pipe_sum, fold_timings, translator):
    folder_text = dict()
    videos = glob.glob(os.path.join(folder, '*.mp4'))
    for i, p in enumerate(videos):
        if fold_timings[i] > 2:
            frames = read_video(p, frames_num=256)
            texts = question_set_inf(frames=frames, model=model)
            text = translator.translate(pipe_sum(" ".join(texts), max_length=int(round((fold_timings[i] - 1) * 18, 0))))
            folder_text[p] = text
    return folder_text

def prune_video(video:np.ndarray, frames_num:int) -> np.ndarray:

# def cut_by_scenes(timecodes:List[Any], video:np.ndarray, fps:int, **prune_args)->None:
#     for start, end in timecodes:
#         if int(end) - int(start) < 4:
#             pass
#         else:
#             frame_pos_start = fps*int(start)
#             frame_pos_end = fps*int(end)








            

   

