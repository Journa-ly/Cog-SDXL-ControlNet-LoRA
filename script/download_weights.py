import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import AutoencoderKL, DiffusionPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

# Load environment variables from the .env file
load_dotenv()
login()

CONTROL_CACHE = "control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"


# Directories and Constants (loaded from environment variables)
AZURE_ACCOUNT_URL = os.getenv("AZURE_ACCOUNT_URL")
JOURNA_CONTAINER_NAME = os.getenv("JOURNA_CONTAINER_NAME")
JOURNA_BLOB_NAME = os.getenv("JOURNA_BLOB_NAME")
JOURNA_MODEL_LOCAL_PATH = os.getenv("JOURNA_MODEL_LOCAL_PATH")
SAS_TOKEN = os.getenv("SAS_TOKEN")

# # Journa Azure Storage URL and Blob Information
# JOURNA_CONTAINER_NAME = ""
# JOURNA_BLOB_NAME = ""
# AZURE_ACCOUNT_URL = ""
# SAS_token = ""

# Function to get BlobServiceClient
def get_blob_service_client_sas(sas_token: str) -> BlobServiceClient:
    logging.info("Creating BlobServiceClient with SAS token.")
    return BlobServiceClient(account_url=AZURE_ACCOUNT_URL, credential=sas_token)

# Function to download blob to a file
def download_blob_to_file(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, download_path: str):
    logging.info(f"Starting download: {container_name}/{blob_name} to {download_path}")
    
    # Create blob client
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    # Create local directory if not exists
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    logging.info(f"Directory {os.path.dirname(download_path)} created/existed already.")

    # Download the blob to the specified file
    with open(download_path, "wb") as file:
        download_stream = blob_client.download_blob()
        logging.info(f"Downloading blob {blob_name}.")
        file.write(download_stream.readall())
        logging.info(f"Download completed for blob {blob_name}.")

# Get the blob service client
blob_service_client = get_blob_service_client_sas(SAS_TOKEN)

# Download the Journa model
download_blob_to_file(
    blob_service_client=blob_service_client,
    container_name=JOURNA_CONTAINER_NAME,
    blob_name=JOURNA_BLOB_NAME,
    download_path=SDXL_MODEL_CACHE
)

logging.info("Downloading additional models.")


better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(REFINER_MODEL_CACHE, safe_serialization=True)

safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float16,
)
safety.save_pretrained(SAFETY_CACHE)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
#controlnet.save_pretrained(CONTROL_CACHE)
