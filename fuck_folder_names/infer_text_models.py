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

    def write_voice(self, texts, speaker='xenia', sample_rate=48000, ssml=False, put_accent=True, put_yo=True,
                    save=False,
                    output_path='out_audio.wav', format='wav', bits_per_sample=64, encoding="PCM_S"):
        '''
        speakers   :  'aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'
        sample_rate:  8000, 24000, 48000
        '''

        for i in range(len(texts)):
            texts[i] = self.num_to_words(texts[i])
            texts[i] = self.to_ssml(texts[i])

            if ssml:
                audio = self.audio_model.apply_tts(ssml_text=texts[i],
                                                   speaker=speaker,
                                                   sample_rate=sample_rate,
                                                   put_accent=put_accent,
                                                   put_yo=put_yo)
            else:
                audio = self.audio_model.apply_tts(text=texts[i],
                                                   speaker=speaker,
                                                   sample_rate=sample_rate,
                                                   put_accent=put_accent,
                                                   put_yo=put_yo)

            if save:
                torchaudio.save(filepath=os.path.join(output_path, f'audio_{i}.wav'),
                                format=format,
                                src=audio[None, :],
                                sample_rate=sample_rate,
                                encoding=encoding,
                                bits_per_sample=bits_per_sample)




class Translator:
    def __init__(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.translate_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device)

    def translate(self, text, **kwargs):

        self.translate_model.eval()
        inputs = self.translate_model(text, return_tensors='pt')
        with torch.no_grad():
            hypotheses = self.translate_model.generate(**inputs, **kwargs)
        return self.translate_model.decode(hypotheses[0], skip_special_tokens=True)
    



