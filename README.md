# Speaker Diarization

Won Techsmith award at Spartahack https://devpost.com/software/speaker-diarization

## Inspiration
The inspiration for this project stemmed from the desire to enhance the efficiency and utility of virtual meetings, particularly in platforms like Zoom. Recognizing the challenges of tracking speakers in dynamic group discussions, the goal was to leverage computer vision to automate speaker diarization. The visual cue of a green box around the speaker's name in real-time, combined with audio synchronization and transcription, was envisioned as a powerful way to seamlessly organize and summarize virtual conversations. The inspiration lies in facilitating better communication, comprehension, and knowledge retention in the era of remote collaboration.

## What it does
The project focuses on automating the process of speaker diarization in Zoom video calls using computer vision. It employs advanced algorithms to identify and associate each speaker with precise timestamps, visually represented by a dynamic green box around their name during the call. This visual mapping is then synchronized with the corresponding audio, which is subsequently transcribed into text. The end result is a streamlined and automated method of converting virtual conversations into written form. Finally, these transcripts are used to generate insightful summaries, providing a comprehensive and user-friendly approach to understanding and archiving discussions in virtual meetings.

## How we built it
The project is a sophisticated integration of computer vision and natural language processing (NLP) techniques to automate speaker diarization in Zoom video calls. The computer vision aspect employs contour detection algorithms, specifically utilizing OpenCV for identifying and tracking speakers in real-time. This is visually represented by a dynamic green box around the speaker's name during the call, establishing a precise timestamp.

Simultaneously, the NLP component leverages the powerful Facebook BART model for audio transcription. The identified speakers are synchronized with the corresponding audio streams, allowing for accurate conversion of spoken content into written text. The BART model excels at generating coherent and contextually relevant text, making it ideal for transcribing the audio and subsequently summarizing the conversation.

The end result is a seamlessly integrated system that combines visual cues with textual representations. Users benefit from an automated and user-friendly solution, facilitating not only speaker identification but also the creation of insightful summaries for efficient comprehension and archival of virtual discussions. The project continuously optimizes accuracy through machine learning algorithms and user feedback, ensuring a robust and evolving solution for speaker diarization in virtual meetings.

