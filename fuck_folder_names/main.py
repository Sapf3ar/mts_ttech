from utils import *


<<<<<<< HEAD
def main(path_to_mkv_file):

    path_to_folder = path_to_mkv_file[:path_to_mkv_file.rfind('/')]
    film_name = path_to_mkv_file.split('/')[-1].split('.')[0]

    path_to_file_for_output_audio = path_to_folder + '/audio/main_audio.wav'
    path_to_audio_folder = path_to_file_for_output_audio[:path_to_file_for_output_audio.rfind('/')]

    os.mkdir(path_to_audio_folder)
    get_audio(input_path=path_to_mkv_file, output_path=path_to_file_for_output_audio)



    get_srt(path=path_to_folder)

    path_to_subtitles = path_to_folder + '/' + film_name + '_subtitles_1.srt'
    free_timings_sec = get_timings(srt_path=path_to_subtitles, in_seconds=True, free=True)

    path_to_cut_videos = path_to_folder + '/cutted_by_timings'
    cut_by_timings(path=path_to_mkv_file, timings=free_timings_sec, output_folder_path=path_to_cut_videos)
=======



def main(path_to_mkv_file):

    path_to_folder = path_to_mkv_file[:path_to_mkv_file.rfind('/')]
    film_name = path_to_mkv_file.split('/')[-1].split('.')[0]

    path_to_file_for_output_audio = path_to_folder + '/audio/main_audio.wav'
    path_to_audio_folder = path_to_file_for_output_audio[:path_to_file_for_output_audio.rfind('/')]

    os.mkdir(path_to_audio_folder)
    get_audio(input_path=path_to_mkv_file, output_path=path_to_file_for_output_audio)



    get_srt(path=path_to_folder)

    path_to_subtitles = path_to_folder + '/' + film_name + '_subtitles_1.srt'
    free_timings_sec = get_timings(srt_path=path_to_subtitles, in_seconds=True, free=True)
    #cut video by free timings then cut into scenes
    #


>>>>>>> 8fb8232 (check this inference)
    """
    your piece of code
    """
    #texts = []
    #timings = []

    os.mkdir(path_to_folder + '/generated_audios')
    path_to_path_with_generated_audios = path_to_folder + '/generated_audios'
<<<<<<< HEAD
    text2audio(texts=texts, model=audio_model, output_path=path_to_path_with_generated_audios, ssml=True, save=True)
=======
    text2audio_model = Text2Audio()
    text2audio_model.write_voice(texts=texts, output_path=path_to_path_with_generated_audios, ssml=True, save=True)
>>>>>>> 8fb8232 (check this inference)

    audio_merge(main_file=path_to_mkv_file, second_path=path_to_path_with_generated_audios, timings=timings, output_file=path_to_folder + '/audio/main_audio_changed.wav')

    add_audio_to_file(input_file=path_to_mkv_file, audio_file=path_to_folder + '/audio/main_audio_changed.wav', output_file=path_to_folder + '/changed_film.mkv')

if __name__ == '__main__':

    path_to_mkv_file = sys.argv[1]
    main(path_to_mkv_file)

