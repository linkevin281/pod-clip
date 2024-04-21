
from moviepy.editor import AudioClip, AudioFileClip, concatenate_audioclips
import scipy
import re

from transformers import AutoProcessor, BarkModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub

# ==================================================
# HELPER FUNCTIONS (scroll down for main functions)
# ==================================================

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

def create_show_intro_prompt(user_prompt):
    template = """<s>[INST]Turn the user's prompt into a one sentence intro using the template: 'Hi! We've created a podlight for you all about [user's topic]. Enjoy!'

    For example, if the user asks for 'podcast about AI', you would respond with 'Hi! We've created a podlight for you all about AI. Enjoy!'

    User prompt: {user_prompt}[/INST]
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
        {"user_prompt": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    output = chain.invoke(user_prompt)
    output = extract_output(output)
    return output


# Creates one line outro prompt text for the tts model
def create_show_outro_prompt(user_prompt):
    template = """<s>[INST]Turn the user's prompt into a one sentence intro using the template: 'Thanks for listening! We hope you enjoyed hearing about [user's topic]. See you soon!'

    Respond with only the output sentence. For example, if the user asks for 'podcast about AI', you would respond with 'Thanks for listening! We hope you enjoyed hearing about AI. See you soon!'

    User prompt: {user_prompt}[/INST]
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
        {"user_prompt": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    output = chain.invoke(user_prompt)
    output = extract_output(output)
    return output


# ====================================================
# MAIN FUNCTIONS
# ====================================================


# Creates the intro segment for a video clip given the title of the podcast and the subtitle of the chapter
def make_whole_show_intro(user_prompt, model, processor, voice_preset):
    intro_prompt = create_show_intro_prompt(user_prompt)
    inputs = processor(intro_prompt, voice_preset=voice_preset)

    # Run voice generator
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Get audio clip
    sample_rate = model.generation_config.sample_rate
    output_filepath = 'podlight_intro_clip.wav'
    scipy.io.wavfile.write(output_filepath, rate=sample_rate, data=audio_array)
    audio_clip = AudioFileClip(output_filepath)  # creating an AudioCLipo from the audio array was super finicky so this was a workaround
    return audio_clip


# Creates the intro segment for a video clip given the title of the podcast and the subtitle of the chapter
def make_clip_intro(index, podcast, chapter_description, model, processor, voice_preset):
    text_prompt = create_clip_intro_prompt(podcast, chapter_description)
    inputs = processor(text_prompt, voice_preset=voice_preset)

    # Run voice generator
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Create audio clip
    sample_rate = model.generation_config.sample_rate
    output_filepath = str(index) + '_intro.wav'
    scipy.io.wavfile.write(output_filepath, rate=sample_rate, data=audio_array)
    output_clip = AudioFileClip(output_filepath)  # creating an AudioCLipo from the audio array was super finicky so this was a workaround
    return output_clip


# Creates the outro segment for a video clip 
def make_outro_clip(user_prompt, model, processor, voice_preset):
    outro_prompt = create_show_outro_prompt(user_prompt)
    inputs = processor(outro_prompt, voice_preset=voice_preset)

    # Run voice generator
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    # Get audio clip
    sample_rate = model.generation_config.sample_rate
    output_filepath = 'podlight_outro_clip.wav'
    scipy.io.wavfile.write(output_filepath, rate=sample_rate, data=audio_array)
    audio_clip = AudioFileClip(output_filepath)  # creating an AudioCLipo from the audio array was super finicky so this was a workaround
    return audio_clip


def concatenate_clips(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and saves it to `output_path`. The extension (e.g., mp3, wav) must be correct in `output_path`."""
    clips = [AudioFileClip(c) for c in audio_clip_paths]  # Load audio clips
    final_clip = concatenate_audioclips(clips)  # Concatenate audio clips
    final_clip.write_audiofile(output_path)  # Save the concatenated clip to the output path
