import re
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, MT5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration

import CONSTANTS as CONSTANTS

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent, MT5ForConditionalGenerationLatent



def get_latent_model(args):
    if 'bart' in args.enc_dec_model:
        config = BartForConditionalGeneration.from_pretrained(
            args.enc_dec_model).config
        config.max_position_embeddings = max(args.max_seq_len, config.max_position_embeddings)
        config.vocab_size=args.vocab_size
        lm = BARTForConditionalGenerationLatent(
            config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents)
    elif 't5' in args.enc_dec_model:
        if 'mt5' in args.enc_dec_model:
            config = MT5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            config.max_position_embeddings = max(args.max_seq_len, config.max_position_embeddings)
            config.vocab_size=args.vocab_size
            lm = MT5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
        else:
            config = T5ForConditionalGeneration.from_pretrained(
                args.enc_dec_model).config
            config.max_position_embeddings = max(args.max_seq_len, config.max_position_embeddings)
            config.vocab_size=args.vocab_size
            lm = T5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
    else:
        print("Unsupported model")
        raise NotImplementedError
    
    if args.lm_mode == 'ft':
        for (param_name, param) in lm.named_parameters():
            param.requires_grad = True
    elif args.lm_mode == 'freeze':
        for (param_name, param) in lm.named_parameters():
            if re.fullmatch(".*perceiver.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError
        
    return lm, config