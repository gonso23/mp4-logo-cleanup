# mp4-logo-cleanup


## Video Splitter Script "split_video.py"

This Python script splits a video file into a specified number of equal-length parts 
The resulting video segments are saved in a folder named `video_parts`.

- Automatically creates output directory and names parts sequentially
- used to slice bigger videos to smaller parts to ease processing

### Usage

```

python split_video.py <video_file_path> <number_of_parts>

```

**Example:**
```

python split_video.py input.mp4 4

```
