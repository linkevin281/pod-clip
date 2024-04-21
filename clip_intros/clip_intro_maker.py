import os
from transformers import AutoProcessor, BarkModel
from moviepy.editor import AudioClip, AudioFileClip, concatenate_audioclips
import scipy
import pandas as pd
from tqdm import tqdm
import re
import torch


from transformers import AutoProcessor, BarkModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function that extracts output from mistral code
def extract_output(text):
    pattern = r"\[/INST\]\s*\n\s*(.*)"

    match = re.search(pattern, text)

    if match:
        user_prompt = match.group(1).strip()
        output = user_prompt
    else:
        output = "Hi! Welcome to your custom podlight. Enjoy!"
    return output
def create_clip_intro_prompt(podcast, chapter_description):
    template = """<s>[INST] Clean up the prompt sentence, adding/removing transition words around the chapter description to make the sentence clearer. Only respond with the output sentence. The prompts should all have the template "Coming up, we've got a clip about [chapter-description] from [podcast-name]. Enjoy!"

    For example, if the chapter description is "Big Tech's involvement in political process' from the 'a16z' podcast, you would respond with 'Coming up, we've got a clip about big tech's involvement in the political process from a16z. Enjoy!"

    For example, if the chapter description is "Thoughts on recent TikTok legislation' from the 'All-In' podcast, you would respond with 'Coming up, we've got a clip about the recent Tiktok legislation from All-In. Enjoy!"

    For example, if the chapter description is "Q: Would either of you ever consider running for office?' from the 'VC20' podcast, you would respond with 'Coming up, we've got a clip about running for office from All-In. Enjoy!"

    Chapter description: {chapter_description}
    Podcast: {podcast}
    [/INST]
    """
    prompt = ChatPromptTemplate.from_template(template)

    # TRAINING LOOP
    model = HuggingFaceHub(
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 30,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )

    chain = (
        {"chapter_description": RunnablePassthrough(), "podcast": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    output = chain.invoke({"chapter_description": chapter_description, "podcast":podcast})
    output = extract_output(output)
    return output

# Creates the intro segment for a video clip given the title of the podcast and the subtitle of the chapter
def make_clip_intro(index, podcast, chapter_description, model, processor, voice_preset, device):
    text_prompt = create_clip_intro_prompt(podcast, chapter_description)
    inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

    # Run voice generator
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Create audio clip
    sample_rate = model.generation_config.sample_rate
    output_filepath = str(index) + '_intro.wav'
    scipy.io.wavfile.write(output_filepath, rate=sample_rate, data=audio_array)
    output_clip = AudioFileClip(output_filepath)  # creating an AudioCLipo from the audio array was super finicky so this was a workaround
    return output_clip

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'TOKEN'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)

voice_preset = "v2/en_speaker_6"
df = pd.read_csv("/home/ubuntu/InferenceTest1/hack24s/pod-clip/clip_intros/true_data.csv")
print(df.shape)
df.head()

# Make a clip intro for each clip
for index, row in tqdm(df.iterrows()):
    make_clip_intro(row['index'], row['channel'], row['chapter'], model, processor, voice_preset, device)

print("all done!")
