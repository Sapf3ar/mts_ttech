from utils import *


def main(way_to_mkv_file):

    way_to_folder = way_to_mkv_file[:way_to_mkv_file.rfind('/')]
    film_name = way_to_mkv_file.split('/')[-1].split('.')[0]

    way_to_file_for_output_audio = way_to_folder + '/audio/main_audio.wav'
    way_to_audio_folder = way_to_file_for_output_audio[:way_to_file_for_output_audio.rfind('/')]

    os.mkdir(way_to_audio_folder)
    get_audio(input_path=way_to_mkv_file, output_path=way_to_file_for_output_audio)



    get_srt(path=way_to_folder)

    way_to_subtitles = way_to_folder + '/' + film_name + '_subtitles_1.srt'
    free_timings_sec = get_timings(srt_path=way_to_subtitles, in_seconds=True, free=True)
    """
    your piece of code
    """
    #texts = []
    #timings = []

    os.mkdir(way_to_folder + '/generated_audios')
    way_to_path_with_generated_audios = way_to_folder + '/generated_audios'
    text2audio(texts=texts, model=audio_model, output_path=way_to_path_with_generated_audios, ssml=True, save=True)

    audio_merge(main_file=way_to_mkv_file, second_path=way_to_path_with_generated_audios, timings=timings, output_file=way_to_folder + '/audio/main_audio_changed.wav')

    add_audio_to_file(input_file=way_to_mkv_file, audio_file=way_to_folder + '/audio/main_audio_changed.wav', output_file=way_to_folder + '/changed_film.mkv')

if __name__ == '__main__':

    way_to_mkv_file = sys.argv[1]
    main(way_to_mkv_file)

