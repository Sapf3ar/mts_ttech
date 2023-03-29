import torch
import numpy as np
import cv2
from typing import Dict, Any
from openvino.runtime import Core
import os
from typing import List, Tuple, Dict
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import BlipProcessor

import py7zr

class BlipEngine:
    """
    Model class for inference BLIP model with OpenVINO
    """
    def __init__(self, main_path:str):
        """
        Initialization class parameters

        """
        print('Initialiazing model...')
        self.load_models(main_path=main_path)
        
        self.vision_model_out = self.vision_model.output(0)
        
        self.text_encoder_out = self.text_encoder.output(0)
        
        # self.config = config
        self.decoder_start_token_id = 30522 
        self.decoder_input_ids = 30522
        self.processor = BlipProcessor.from_pretrained(os.path.join(main_path, 'config'))

    def get_weights(self, path:str) -> str:
        if "7z" in path:
            if os.path.exists(path.split('.')[0]) and os.path.isdir(path.split('.')[0]):
                return path.split('.')[0]
            
            with py7zr.SevenZipFile(path, 'r') as archive:
                print(f"Extracting archive to {path.split('.')[0]}")
                archive.extractall(path=".")
            return path.split('.')[0]
        return path
    
    def load_models(self, main_path:str) -> Dict[str, Any]:
        main_path = self.get_weights(main_path)

        ie = Core() #create inference engine
        paths = [
            'blip_text_encoder.onnx',
            'blip_text_decoder.onnx',
            'blip_text_decoder_with_past.onnx',
            "blip_vision_model.onnx"
        ]
        model_onnx = ie.read_model(model=os.path.join(main_path, paths[0]))
        encoder_engine = ie.compile_model(model=model_onnx, device_name="CPU")
        print("Encoder loaded...")
        model_onnx = ie.read_model(model=os.path.join(main_path, paths[1]))
        decoder_engine = ie.compile_model(model=model_onnx,  device_name="CPU")
        print("Decoder loaded...")
        # model_onnx = ie.read_model(model=os.path.join(main_path, paths[2]))
        # decoder_qa_engine = ie.compile_model(model=model_onnx = ie.read_model(model=os.path.join(main_path, paths[0])), device_name="GPU")

        model_onnx = ie.read_model(model=os.path.join(main_path, paths[3]))
        visual_engine = ie.compile_model(model=model_onnx, device_name="CPU")
        print("Visual model loaded..")
        print("All model loaded..")
        self.vision_model = visual_engine
        self.text_encoder = encoder_engine
        self.text_decoder = decoder_engine

    
    def generate_answer(self, pixel_values:torch.Tensor, input_ids:torch.Tensor, attention_mask:torch.Tensor, **generate_kwargs):
        """
        Visual Question Answering prediction
        Parameters:
          pixel_values (torch.Tensor): preprocessed image pixel values
          input_ids (torch.Tensor): question token ids after tokenization
          attention_mask (torch.Tensor): attention mask for question tokens
        Retruns:
          generation output (torch.Tensor): tensor which represents sequence of generated answer token ids
        """
        # image_embed = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]

        flag = True
        for frame in pixel_values:
                if flag:
                
                    frames_embs  = self.vision_model(frame.detach().numpy())[self.vision_model_out]
                    flag = False
                else:
                    embs = self.vision_model(frame.detach().numpy())[self.vision_model_out]
                    frames_embs = np.concatenate((frames_embs, embs), axis=1)

        image_attention_mask = np.ones(frames_embs.shape[:-1], dtype=int)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        question_embeds = self.text_encoder([input_ids.detach().numpy(), attention_mask.detach().numpy(), frames_embs, image_attention_mask])[self.text_encoder_out]
        question_attention_mask = np.ones(question_embeds.shape[:-1], dtype=int)

        bos_ids = np.full((question_embeds.shape[0], 1), fill_value=self.decoder_start_token_id)

        outputs = self.text_decoder.generate(
            input_ids=torch.from_numpy(bos_ids),
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=torch.from_numpy(question_embeds),
            encoder_attention_mask=torch.from_numpy(question_attention_mask),
            **generate_kwargs,
        )
        return outputs

    def generate_caption(self, pixel_values:List[torch.Tensor], input_ids:torch.Tensor = None, attention_mask:torch.Tensor = None, **generate_kwargs):
        """
        Image Captioning prediction
        Parameters:
          pixel_values (torch.Tensor): preprocessed image pixel values
          input_ids (torch.Tensor, *optional*, None): pregenerated caption token ids after tokenization, if provided caption generation continue provided text
          attention_mask (torch.Tensor): attention mask for caption tokens, used only if input_ids provided
        Retruns:
          generation output (torch.Tensor): tensor which represents sequence of generated caption token ids
        """
        batch_size = pixel_values.shape[0]

        # image_embeds = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]
        #concantenate embeddings of video frames
        flag = True
        for frame in pixel_values:
                if flag:
                
                    frames_embs  = self.vision_model(frame.detach().numpy())[self.vision_model_out]
                    flag = False
                else:
                    embs = self.vision_model(frame.detach().numpy())[self.vision_model_out]
                    frames_embs = np.concatenate((frames_embs, embs), axis=1)

        image_attention_mask = torch.ones(frames_embs.shape[:-1], dtype=torch.long)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
            )
        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=torch.from_numpy(frames_embs),
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs

    def caption(self, raw_image, **generate_kwargs) -> str:
        inputs = self.processor(raw_image, ' ', return_tensors='pt')
        out = self.generate_caption(inputs["pixel_values"], **generate_kwargs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def answer(self,raw_image, question, **generate_kwargs)->str:
        inputs = self.processor(raw_image, question, return_tensors='pt')
        out = self.generate_answer(**inputs, **generate_kwargs)
        return out

 