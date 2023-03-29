import torch
from omegaconf import OmegaConf
import torchaudio
import num2words

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Text2Audio:
    def __init__(self) -> None:
        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                       'latest_silero_models.yml',
                                       progress=False)
        OmegaConf.load('latest_silero_models.yml')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                        model='silero_tts',
                                                        language='ru',
                                                        speaker='v3_1_ru')
        self.audio_model.to(device)

    def to_ssml(self, text):
        start = '''
                  <speak>
                  <p>
                '''

        end = '''
                </p>
                </speak>
              '''

        return start + text + end

    def num_to_words(self, text):
        after_spliting = text.split()

        for index in range(len(after_spliting)):
            if after_spliting[index].isdigit():
                after_spliting[index] = num2words(after_spliting[index], lang='ru')
        numbers_to_words = ' '.join(after_spliting)
        return numbers_to_words


    def get_timings_for_audio(self, timecodes):
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
            # time = time_end - time_start
            fold_timings.append([time_start, time_end])
        return fold_timings

    def write_voice(self, texts, local_timings, speaker='xenia', sample_rate=48000, ssml=False, put_accent=True,
                    put_yo=True,

    def write_voice(self, texts, local_timings, speaker='xenia', sample_rate=48000, ssml=False, put_accent=True, put_yo=True,
                    save=False,
                    output_path='out_audio.wav', format='wav', bits_per_sample=64, encoding="PCM_S"):
        '''
        speakers   :  'aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'
        sample_rate:  8000, 24000, 48000
        '''


        for i, folder in enumerate(texts):

        for folder in texts:
            e2 = folder.split('/')[-1].split('_')[-1]
            e1 = folder.split('/')[-1].split('_')[-2]
            s2 = folder.split('/')[-1].split('_')[-3]
            s1 = folder.split('/')[-1].split('_')[-4]

            start = float(s1 + "." + s2) - 0.5
            end = float(e1 + "." + e2)
            scene_timcodes = self.get_timings_for_audio(local_timings[i])
            for i, id in enumerate(texts[folder]):
                text = texts[folder][id][1]
                # print(start)
                # print(scene_timcodes[i])
                start += scene_timcodes[i][0]
                # print(start)
                s1 = str(start).split('.')[0]
                s2 = str(start).split('.')[1]
                # print('-'*80)

                try:
                    text = self.num_to_words(text)
                    # texts[i] = self.to_ssml(texts[i])
                    if ssml:
                        audio = self.audio_model.apply_tts(ssml_text=text,
                                                           speaker=speaker,
                                                           sample_rate=sample_rate,
                                                           put_accent=put_accent,
                                                           put_yo=put_yo)
                    else:
                        audio = self.audio_model.apply_tts(text=text,
                                                           speaker=speaker,
                                                           sample_rate=sample_rate,
                                                           put_accent=put_accent,
                                                           put_yo=put_yo)

                    if save:
                        torchaudio.save(filepath=os.path.join(output_path, f'audio_{s1}_{s2}.wav'),
                                        format=format,
                                        src=audio[None, :],
                                        sample_rate=sample_rate,
                                        encoding=encoding,
                                        bits_per_sample=bits_per_sample)
                except:
                    continue

            #start = float(s1 + "." + s2)
            #end   = float(e1 + "." + e2)
            for id in texts[folder]:
                text = texts[folder][id]
                text = self.num_to_words(text)
                #texts[i] = self.to_ssml(texts[i])

                if ssml:
                    audio = self.audio_model.apply_tts(ssml_text=text,
                                                      speaker=speaker,
                                                      sample_rate=sample_rate,
                                                      put_accent=put_accent,
                                                      put_yo=put_yo)
                else:
                    audio = self.audio_model.apply_tts(text=text,
                                                      speaker=speaker,
                                                      sample_rate=sample_rate,
                                                      put_accent=put_accent,
                                                      put_yo=put_yo)

                if save:
                    torchaudio.save(filepath=os.path.join(output_path, f'audio_{s1}_{s2}_{e1}_{e2}.wav'),
                                    format=format,
                                    src=audio[None, :],
                                    sample_rate=sample_rate,
                                    encoding=encoding,
                                    bits_per_sample=bits_per_sample)





class Translator:
    def __init__(self) -> None:


        self.translate_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    def translate(self, text, **kwargs):


        self.translate_model.eval()
        inputs = self.translate_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            hypotheses = self.translate_model.generate(**inputs, **kwargs)
        return self.translate_tokenizer.decode(hypotheses[0], skip_special_tokens=True)



