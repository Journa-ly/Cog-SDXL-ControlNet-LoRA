import argparse
import os
import shutil
import logging
from dotenv import load_dotenv
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from the .env file
load_dotenv()

# Retrieve the token from the environment variables
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    logging.error("HF_TOKEN is not set. Please ensure it's specified in your environment.")
else:
    try:
        login(token=hf_token)
        logging.info("Login succeeded")
    except HfHubLoginError as e:
        logging.error(f"Login failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    Swin2SRForImageSuperResolution,
)

DEFAULT_BLIP = "Salesforce/blip-image-captioning-large"
DEFAULT_CLIPSEG = "CIDAS/clipseg-rd64-refined"
DEFAULT_SWINIR = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
    blip_processor = BlipProcessor.from_pretrained(DEFAULT_BLIP)
    blip_model = BlipForConditionalGeneration.from_pretrained(DEFAULT_BLIP)

    clip_processor = CLIPSegProcessor.from_pretrained(DEFAULT_CLIPSEG)
    clip_model = CLIPSegForImageSegmentation.from_pretrained(DEFAULT_CLIPSEG)

    swin_model = Swin2SRForImageSuperResolution.from_pretrained(DEFAULT_SWINIR)    

    temp_models = 'tmp/models'
    if os.path.exists(temp_models):
        shutil.rmtree(temp_models)
    os.makedirs(temp_models)

    blip_processor.save_pretrained(os.path.join(temp_models, 'blip_processor'))
    blip_model.save_pretrained(os.path.join(temp_models, 'blip_large'))
    clip_processor.save_pretrained(os.path.join(temp_models, 'clip_seg_processor'))
    clip_model.save_pretrained(os.path.join(temp_models, 'clip_seg_rd64_refined'))
    swin_model.save_pretrained(os.path.join(temp_models, 'swin2sr_realworld_sr_x4_64_bsrgan_psnr'))

    for val in os.listdir(temp_models):
        if 'tar' not in val:
            os.system(f'sudo tar -cvf {os.path.join(temp_models, val)}.tar -C {os.path.join(temp_models, val)} .')
            os.system(f'gcloud storage cp -R {os.path.join(temp_models, val)}.tar gs://{args.bucket}/{val}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", "-m", type=str)
    args = parser.parse_args()
    upload(args)
