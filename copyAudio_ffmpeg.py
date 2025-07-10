import subprocess
import os

def extract_audio(video_path, audio_path):
    """
    Extracts the audio track from a video file and saves it as a separate file.

    :param video_path: Path to the input video file.
    :param audio_path: Path to the output audio file (e.g., audio.aac or audio.mp3).
    :raises FileNotFoundError: If the input video file does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Prepare ffmpeg command to extract audio without re-encoding
    cmd = [
        "ffmpeg", "-y",               # Overwrite output files without asking
        "-i", video_path,             # Input video file
        "-vn",                        # Disable video recording (extract audio only)
        "-acodec", "copy",            # Copy audio stream without re-encoding
        audio_path                    # Output audio file
    ]
    subprocess.run(cmd, check=True)
    print(f"Audio extracted to: {audio_path}")

def merge_audio(video_path, audio_path, output_path):
    """
    Merges an audio track with a video file (useful if the video has no sound).

    :param video_path: Path to the input video file (without audio or with unwanted audio).
    :param audio_path: Path to the input audio file.
    :param output_path: Path to the output video file with merged audio.
    :raises FileNotFoundError: If either the video or audio file does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    # Prepare ffmpeg command to merge video and audio
    cmd = [
        "ffmpeg", "-y",                   # Overwrite output files without asking
        "-i", video_path,                 # Input video file
        "-i", audio_path,                 # Input audio file
        "-c:v", "copy",                   # Copy video stream without re-encoding
        "-c:a", "aac",                    # Encode audio as AAC for compatibility
        "-shortest",                      # Output duration matches the shortest input
        output_path                       # Output video file
    ]
    subprocess.run(cmd, check=True)
    print(f"Audio and video merged into: {output_path}")

if __name__ == "__main__":
    # Step 1: Extract audio from the original video
    src_vid = "../input.mp4"
    base_name = os.path.splitext(os.path.basename(src_vid))[0]
    extracted_audio = f"{base_name}.aac"
    
    extract_audio(src_vid, extracted_audio)
    
    # Step 2: Merge the extracted audio with another video file
    dst_video = "output.mp4"
    base, ext = os.path.splitext(dst_video)  # base = "output", ext = ".mp4"
    out_video = f"{base}_audio{ext}"      # ergibt "output_audio.mp4"
    merge_audio(dst_video, extracted_audio, out_video)
