import torch
import numpy as np
import cv2
from typing import Dict, Any
from openvino.runtime import Core
import os
from typing import List, Tuple, Dict
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from transformers import BlipProcessor, BlipForQuestionAnswering

import py7zr



def prepare_past_inputs(past_key_values:List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Helper function for rearrange input hidden states inputs to OpenVINO model expected format
    Parameters:
      past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): list of pairs key, value attention hidden states obtained as model outputs from previous step
    Returns:
      inputs (Dict[str, torch.Tensor]): dictionary with inputs for model
    """
    inputs = {}
    for idx, (key, value) in enumerate(past_key_values):
        inputs[f"in_past_key_value.{idx}.key"] = key
        inputs[f"in_past_key_value.{idx}.value"] = value
    return inputs

past_key_values_outs = ['out_past_key_value.0.key',
 'out_past_key_value.0.value',
 'out_past_key_value.1.key',
 'out_past_key_value.1.value',
 'out_past_key_value.2.key',
 'out_past_key_value.2.value',
 'out_past_key_value.3.key',
 'out_past_key_value.3.value',
 'out_past_key_value.4.key',
 'out_past_key_value.4.value',
 'out_past_key_value.5.key',
 'out_past_key_value.5.value',
 'out_past_key_value.6.key',
 'out_past_key_value.6.value',
 'out_past_key_value.7.key',
 'out_past_key_value.7.value',
 'out_past_key_value.8.key',
 'out_past_key_value.8.value',
 'out_past_key_value.9.key',
 'out_past_key_value.9.value',
 'out_past_key_value.10.key',
 'out_past_key_value.10.value',
 'out_past_key_value.11.key',
 'out_past_key_value.11.value']
def postprocess_text_decoder_outputs(output:Dict):
    global past_key_values_outs
    """
    Helper function for rearranging model outputs and wrapping to CausalLMOutputWithCrossAttentions
    Parameters:
      output (Dict): dictionary with model output
    Returns
      wrapped_outputs (CausalLMOutputWithCrossAttentions): outputs wrapped to CausalLMOutputWithCrossAttentions format
    """
    outs = {k.any_name: v for k, v in output.items()}
    logits = torch.from_numpy(outs["logits"])
    past_kv = []
    for i in range(0, len(past_key_values_outs), 2):
        key = past_key_values_outs[i]
        value = key.replace(".key", ".value")
        past_kv.append((torch.from_numpy(outs[key]), torch.from_numpy(outs[value])))
    return CausalLMOutputWithCrossAttentions(
        loss=None,
        logits=logits,
        past_key_values=past_kv,
        hidden_states=None,
        attentions=None,
        cross_attentions=None
    )




class BlipEngine:
    """
    Model class for inference BLIP model with OpenVINO
    """
    def __init__(self, main_path:str):
        """
        Initialization class parameters

        """
        
        print('Initialiazing model...')
        self.text_decoder = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").text_decoder
        self.load_models(main_path=main_path)
        
        self.vision_model_out = self.vision_model.output(0)
        

        self.text_encoder_out = self.text_encoder.output(0)
        self.sep_token_id = 102
        self.text_config  = {
            "attention_probs_dropout_prob": 0.0,
            "bos_token_id": 30522,
            "encoder_hidden_size": 768,
            "eos_token_id": 2,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "is_decoder": True,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "blip_text_model",
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "projection_dim": 768,
            "sep_token_id": 102,
            "transformers_version": "4.28.0.dev0",
            "use_cache": True,
            "vocab_size": 30524
            }
        # self.config = config

        self.decoder_start_token_id = 30522 
        self.decoder_input_ids = 30522


    def text_decoder_forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, past_key_values:List[Tuple[torch.Tensor, torch.Tensor]], encoder_hidden_states:torch.Tensor, encoder_attention_mask:torch.Tensor, **kwargs):
        """
        Inference function for text_decoder in one generation step
        Parameters:
        input_ids (torch.Tensor): input token ids
        attention_mask (torch.Tensor): attention mask for input token ids
        past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): list of cached decoder hidden states from previous step
        encoder_hidden_states (torch.Tensor): encoder (vision or text) hidden states
        encoder_attention_mask (torch.Tensor): attnetion mask for encoder hidden states
        Returns
        model outputs (CausalLMOutputWithCrossAttentions): model prediction wrapped to CausalLMOutputWithCrossAttentions class including predicted logits and hidden states for caching
        """
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask
        }
        if past_key_values is None:
            outputs = self.text_decoder_ov(input_dict)
        else:
            input_dict.update(prepare_past_inputs(past_key_values))
            outputs = self.text_decoder_with_past(input_dict)
        return postprocess_text_decoder_outputs(outputs)
    
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
        self.processor = BlipProcessor.from_pretrained(os.path.join(main_path, 'config'))
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
        model_onnx = ie.read_model(model=os.path.join(main_path, paths[2]))
        decoder_qa_engine = ie.compile_model(model=model_onnx = ie.read_model(model=os.path.join(main_path, paths[0])), device_name="GPU")

        model_onnx = ie.read_model(model=os.path.join(main_path, paths[3]))
        visual_engine = ie.compile_model(model=model_onnx, device_name="CPU")
        print("Visual model loaded..")
        print("All model loaded..")
        self.vision_model = visual_engine
        self.text_encoder = encoder_engine
        self.text_decoder_ov = decoder_engine
        self.text_decoder_with_past = decoder_qa_engine
        self.text_decoder.forward = self.text_decoder_forward

    
    def generate_answer(self, pixel_values:List[torch.Tensor], input_ids:torch.Tensor, attention_mask:torch.Tensor, **generate_kwargs):
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
                    inputs = self.processor(frame, ' ', return_tensors='pt')['pixel_values']
                    frames_embs  = self.vision_model(inputs.detach().numpy())[self.vision_model_out]
                    flag = False
                else:
                    inputs = self.processor(frame, ' ', return_tensors='pt')['pixel_values']
                    embs = self.vision_model(inputs.detach().numpy())[self.vision_model_out]
                    frames_embs = np.concatenate((frames_embs, embs), axis=1)

        image_attention_mask = np.ones(frames_embs.shape[:-1], dtype=int)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        question_embeds = self.text_encoder([input_ids.detach().numpy(), attention_mask.detach().numpy(), frames_embs, image_attention_mask])[self.text_encoder_out]
        question_attention_mask = np.ones(question_embeds.shape[:-1], dtype=int)

        bos_ids = np.full((question_embeds.shape[0], 1), fill_value=self.decoder_start_token_id)

        outputs = self.text_decoder.generate(
            input_ids=torch.from_numpy(bos_ids),
            eos_token_id=self.text_config['sep_token_id']
            pad_token_id=self.text_config['pad_token_id'],
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
        batch_size = 1

        # image_embeds = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]
        #concantenate embeddings of video frames
        flag = True
        for frame in pixel_values:
                if flag:
                    inputs = self.processor(frame, ' ', return_tensors='pt')['pixel_values']
                    frames_embs  = self.vision_model(inputs.detach().numpy())[self.vision_model_out]
                    flag = False
                else:
                    inputs = self.processor(frame, ' ', return_tensors='pt')['pixel_values']
                    embs = self.vision_model(inputs.detach().numpy())[self.vision_model_out]
                    frames_embs = np.concatenate((frames_embs, embs), axis=1)

        image_attention_mask = torch.ones(frames_embs.shape[:-1], dtype=torch.long)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.text_config['bos_token_id'], self.text_config['eos_token_id']]])
                .repeat(batch_size, 1)
            )
        input_ids[:, 0] = self.text_config['bos_token_id']
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.text_config['sep_token_id']
            pad_token_id=self.text_config['pad_token_id'],
            attention_mask=attention_mask,
            encoder_hidden_states=torch.from_numpy(frames_embs),
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs
        )

        return outputs

    def caption(self, raw_image, **generate_kwargs) -> str:
        # print(f"raw image shape {raw_image.shape}")
        out = self.generate_caption(raw_image, **generate_kwargs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def answer(self,raw_image, question, **generate_kwargs)->str:
        inputs = self.processor(raw_image, question, return_tensors='pt')
        out = self.generate_answer(**inputs, **generate_kwargs)
        return out

 