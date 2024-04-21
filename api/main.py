from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pydub import AudioSegment

import torch, boto3, dotenv, os, asyncio

ENV_PATH = "../.env"
NUM_TOP_K = 5
BUCKET_URL = os.getenv("BUCKET_URL")
BUCKET_NAME = "hackathon24s"
CLIP_S3_PATH = f"test"
MERGED_AUDIO_PATH = "audio/merged"
CUSTOM_AUDIO_PATH = "audio/custom"
SUFFIX = "intro"

dotenv.load_dotenv(ENV_PATH)

########### Things to be declared in highest level api
# import dotenv, os
# from pinecone import Pinecone
# device = "cuda" if torch.cuda.is_available() else "cpu"
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="production")
# pc_index = pc.Index("hackathon24s")
############################

def query_controller(query_text, model):
    """
    Controller function to query the model
        - model: SentenceTransformer model
        - query_text: string to be queried
        - pc_index: Pinecone index
    """
    ## Generate embedding
    with torch.no_grad():
        embedding = model.encode(query_text, convert_to_tensor=True).tolist()
    # fetch pinecone 
    return embedding

async def merge_audio(ids):
    """
    Function to merge all audio files given ids into one"""
    merged_track = AudioSegment.from_file(f"{MERGED_AUDIO_PATH}/{ids[0]}_merge.wav")
    for i in range(1, len(ids)):
        merged_track += AudioSegment.from_file(f"{MERGED_AUDIO_PATH}/{ids[i]}_merged.wav")
    
     # Construct the output filename
    name = f"custom_{'_'.join(ids)}"
    output_file_path = f'{CUSTOM_AUDIO_PATH}/{name}.mp3'

    # Export to a single MP3 file
    merged_track.export(output_file_path, format="mp3")
    await asyncio.sleep(1)

    s3 = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY"), aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))
    s3.upload_file(output_file_path, BUCKET_NAME, f"{CLIP_S3_PATH}/{name}.mp3")
    
    return f"{BUCKET_URL}/{CLIP_S3_PATH}/{name}.mp3"


# Use; for i, (score, channel, chapter) in enumerate(pc_response):
def query_pinecone_from_embedding(embedding, pc_index):
    """
    Function to query pinecone with the generated embedding
        - embedding: torch.tensor
    """
    response = pc_index.query(vector=embedding, top_k=5, include_values=True, include_metadata=True)
    pc_response = []
    for i, match in enumerate(response['matches']):
        metadata = match['metadata']
        score = match['score']
        id = match['id']
        pc_response.append((id, score, metadata['channel'], metadata['chapter']))

    return pc_response