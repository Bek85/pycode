import os
import io
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client
client = OpenAI()


# Function to split a file into chunks of a specific size
def split_file(file_path, chunk_size_bytes=20 * 1024 * 1024):  # Default 20MB chunks
    # Get total file size
    file_size = os.path.getsize(file_path)
    chunks = []

    with open(file_path, "rb") as f:
        # Read and yield chunks
        for i in range(0, file_size, chunk_size_bytes):
            # Read the chunk
            f.seek(i)
            chunk_data = f.read(min(chunk_size_bytes, file_size - i))
            chunks.append(chunk_data)

    return chunks


# Function to transcribe audio chunks
def transcribe_audio(file_path, chunk_size_mb=20):
    # Convert MB to bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Split the file
    print(f"Splitting file into {chunk_size_mb}MB chunks...")
    chunks = split_file(file_path, chunk_size_bytes)
    print(f"File split into {len(chunks)} chunks")

    # Initialize an empty string to store the transcription
    full_transcript = ""

    # Process each chunk
    for i, chunk_data in enumerate(chunks):
        try:
            print(f"Transcribing chunk {i+1}/{len(chunks)}...")

            # Create a file-like object from the bytes
            chunk_file = io.BytesIO(chunk_data)
            chunk_file.name = f"chunk_{i}.mp3"  # Give it a name for OpenAI's API

            # Transcribe the chunk
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=chunk_file
            )

            # Add to full transcript
            full_transcript += transcript.text + " "

        except Exception as e:
            print(f"Error transcribing chunk {i+1}: {e}")

    return full_transcript.strip()


# Main execution
audio_file_path = "disposition_and_sentencing_12-07-2018.mp3"
transcript = transcribe_audio(audio_file_path)
print("\nFull Transcript:")
print(transcript)
