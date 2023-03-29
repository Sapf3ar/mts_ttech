from utils import *
from infer_text_models import Translator, Text2Audio
from transformers import pipeline
from infer import BlipEngine
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Parse training data arguments')

    parser.add_argument('mkv_path', type=str, help='path ro videos')
    parser.add_argument('root_dir', type=str, help='Main dataset dir')
    parser.add_argument('weights_path', type=str, help='Directory to save weights')


    args = parser.parse_args()
    return args


def main(args):
    path_to_mkv_file = args.mkv_path
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

    path_to_folder = path_to_mkv_file[:path_to_mkv_file.rfind('/')]
    film_name = path_to_mkv_file.split('/')[-1].split('.')[0]

    path_to_file_for_output_audio = path_to_folder + '/audio/main_audio.wav'
    path_to_audio_folder = path_to_file_for_output_audio[:path_to_file_for_output_audio.rfind('/')]

    os.mkdir(path_to_audio_folder)
    get_audio(input_path=path_to_mkv_file, output_path=path_to_file_for_output_audio)


    model_engine = BlipEngine(args.weights_path) #path to weights
    get_srt(path=path_to_folder)

    path_to_subtitles = os.path.join(path_to_folder, film_name + '_subtitles_1.srt')

    free_timings_sec = get_timings(srt_path=path_to_subtitles, in_seconds=True, free=True)

    path_to_cut_videos = os.path.join(path_to_folder, '/cutted_by_timings')

    cut_by_timings(path=path_to_mkv_file, timings=free_timings_sec, output_folder_path=path_to_cut_videos)
    relative_inner_timecodes, free_video_folder = global_scene_cut(path_to_cut_videos=path_to_cut_videos)

    all_texts = dict()
    for fold in free_video_folder:
        all_texts[fold] = blip_scene_inf(model_engine, folder=fold,pipe_sum=summarizer)


    

    os.mkdir(path_to_folder + '/generated_audios')
    path_to_path_with_generated_audios = path_to_folder + '/generated_audios'
    silero_model = Text2Audio()
    silero_model.write_voice(texts=texts, output_path=path_to_path_with_generated_audios, ssml=True, save=True)

    audio_merge(main_file=path_to_mkv_file, second_path=path_to_path_with_generated_audios, timings=timings, output_file=path_to_folder + '/audio/main_audio_changed.wav')

    add_audio_to_file(input_file=path_to_mkv_file, audio_file=path_to_folder + '/audio/main_audio_changed.wav', output_file=path_to_folder + '/changed_film.mkv')

if __name__ == '__main__':

    args = get_parser()
    main(args)

