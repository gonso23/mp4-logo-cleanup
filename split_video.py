#!/usr/bin/env python3
import os
import sys
from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_file, n_parts):
    """
    Splits the input video file into n_parts equal segments and saves them in a directory.
    
    Args:
        input_file (str): Path to the input video file.
        n_parts (int): Number of parts to split the video into.
    """
    # Load the video file
    video = VideoFileClip(input_file)
    duration = video.duration
    part_duration = duration / n_parts

    output_dir = "video_parts"
    # Alternatively, save output next to the input file:
    # output_dir = os.path.join(os.path.dirname(os.path.abspath(input_file)), "video_parts")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare base name and extension for output files
    base = os.path.splitext(os.path.basename(input_file))[0]
    ext = os.path.splitext(input_file)[1]
    digits = len(str(n_parts - 1))  # e.g., 10 parts → 1 digit, 100 parts → 2 digits

    for i in range(n_parts):
        # Calculate start and end time for each part
        start_time = i * part_duration
        end_time = (i + 1) * part_duration if i < n_parts - 1 else duration
        # Extract the video subclip
        part = video.subclip(start_time, end_time)
        part_num = f"{i:0{digits}d}"  # e.g., 00, 01, 02, ...
        output_name = f"{base}_P{part_num}{ext}"
        output_path = os.path.join(output_dir, output_name)
        print(f"Creating {output_path} ({start_time:.2f}s – {end_time:.2f}s)")
        # Write the video part to file
        part.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"Video has been split into {n_parts} parts and saved in the '{output_dir}' directory.")

def main():
    """
    Main function to parse command-line arguments and start the video splitting process.
    """
    if len(sys.argv) != 3:
        print("Usage: python split_video.py <video_file_path> <number_of_parts>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        n_parts = int(sys.argv[2])
    except ValueError:
        print("Error: <number_of_parts> must be an integer.")
        sys.exit(1)

    split_video(input_file, n_parts)

if __name__ == "__main__":
    main()
