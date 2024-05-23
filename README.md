# Podlite

**HackDartmouth 2024 Runner-Up**

Podlite is a customized podcast creator that take the best clips from your favorite podcasts and combines them into a single episode. Users can provide an LLM-style prompt to specify the topic of the episode, and Podlite will use a combination of NLP and audio processing to find the best clips that match the prompt.

AI music and text-to-speech is used to introduce each new clip.

## Samples

Prompt: "I want to learn about the current state of AI, especially about Large Language Models."
[Link](https://drive.google.com/file/d/11V4yogl_6QD01XDstMp67bG_NwGTqq5_/view?usp=sharing)

Prompt: "Please make me a podcast about nuclear energy and venture capital."
[Link](https://drive.google.com/file/d/10buY4NJAjJQxMrBGO3cyLckQ3N2LGp6z/view?usp=sharing)

## Stack:

- Frontend: Django
- Backend: Flask
- Database: SQLLite, Pinecone, S3
- AI: Mistral8x7b, Snowflake Arctic Embed, Huggingface, Suno, Google Cloud TTS

![Podlite](assets/podlite.png)

## Architecture

![Architecture](assets/architecture.png)
