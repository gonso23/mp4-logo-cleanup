# mp4-logo-cleanup

## Objective
Remove a logo or something like this from a video

in a nutshell: 
a mask is generated based on a logo or via a manual drawing in the editor and 
the mask is used to remove the logo.

put a logo into the in_logo directory
start with logo_remover_SIFT.py (change the directories in main - sorry for this)
in case it takes long - you should split the video
you can have a look to the cleaned_frames directory to check the mask generation is working
if some of the logos are still there -> use the mask editor to manually update the masks.
finally you need to copy the audio from the src to the output video

## Installation
- Python 3
- use a virtual environment as we do need some old libary versions (python -m venv venv)
- prepare your installation (pip install -r requirements.txt)

## Design
The scripts are some how dependent based on file system

- video_parts -> in case you do process larger videos you may slice it in n parts
- in_logo -> here you do store the logos as png
- frames -> the frames from the input video (generated)
- masks -> the masks generated from the logos or by the manual editor
- cleaned_frames -> the frames where the logo should be removed already

On my Laptop a 5 minutes video takes about 25 minutes to be processed, in the logo_remover_SIFT.py you can switch from single threaded to multithreded and multitasking variants.
For smaller videos this is ok, bigger videos I sliced into parts and executed on different PCs in parallel.

## Video Splitter Script "split_video.py"

This Python script splits a video file into a specified number of equal-length parts 
The resulting video segments are saved in a folder named `video_parts`.

- Automatically creates output directory and names parts sequentially
- used to slice bigger videos to smaller parts to ease processing

### Usage
```
python split_video.py <video_file_path> <number_of_parts>
```

