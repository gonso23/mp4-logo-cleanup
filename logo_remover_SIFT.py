#!/usr/bin/env python3
import cv2
import os
import numpy as np
from glob import glob
import concurrent.futures
import time
import subprocess
import shutil
from multiprocessing import Pool

# 1. Extract frames from the video
def extract_frames(video_path, frame_dir):
    """
    Extracts all frames from a video file and saves them as PNG images in the specified directory.
    """
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    return frame_count

# 2. Load all logo templates (in color)
def load_logo_templates(logo_dir):
    """
    Loads all PNG logo templates from the given directory.
    Returns a list of loaded images.
    """
    templates = []
    for logo_path in glob(os.path.join(logo_dir, "*.png")):
        logo_img = cv2.imread(logo_path)
        if logo_img is not None:
            templates.append(logo_img)
    return templates

# 3. Extract SIFT features from templates
def prepare_template_features(templates):
    """
    Computes SIFT keypoints and descriptors for each template.
    Returns a list of (template, keypoints, descriptors).
    """
    sift = cv2.SIFT_create()
    features = []
    for template in templates:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        kp, des = sift.detectAndCompute(eq, None)
        features.append((template, kp, des))
    return features

# 4. Generate mask for a frame
def generate_mask(frame_path, template_features, mask_dir, match_threshold=10, max_occurrences=5):
    """
    Detects and masks all occurrences of the logos in a frame using SIFT feature matching.
    Saves the mask as a PNG file in the mask directory.
    """
    os.makedirs(mask_dir, exist_ok=True)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    frame = cv2.imread(frame_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = sift.detectAndCompute(gray, None)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if des_f is None or len(kp_f) == 0:
        return None

    for template, kp_t, des_t in template_features:
        if des_t is None or len(kp_t) == 0:
            continue

        matches = bf.knnMatch(des_t, des_f, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Find multiple occurrences of the logo in the frame
        occurrences = 0
        while len(good_matches) >= match_threshold and occurrences < max_occurrences:
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                break

            matches_mask = mask_h.ravel().tolist()
            inliers = [good_matches[i] for i in range(len(matches_mask)) if matches_mask[i]]
            if len(inliers) < match_threshold:
                break

            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.fillConvexPoly(mask, np.int32(dst), 255)

            # Remove inlier matches to find additional occurrences
            inlier_train_idx = set(m.trainIdx for m in inliers)
            good_matches = [m for m in good_matches if m.trainIdx not in inlier_train_idx]
            occurrences += 1

    mask_path = os.path.join(mask_dir, os.path.basename(frame_path))
    cv2.imwrite(mask_path, mask)
    return mask_path

# 5. Inpainting for a frame
def inpaint_frame(frame_path, mask_path, cleaned_dir):
    """
    Uses the mask to remove the logo from the frame via inpainting.
    Saves the cleaned frame in the specified directory.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    output_path = os.path.join(cleaned_dir, os.path.basename(frame_path))
    cv2.imwrite(output_path, inpainted)
    return output_path

# 6. Main processing loop
## Iterative version: processes frames one by one
def process_all_frames_single(frames_dir, templates, mask_dir, cleaned_dir, match_threshold=10):
    """
    Processes all frames sequentially: generates masks and inpaints each frame.
    """
    frame_paths = sorted(glob(os.path.join(frames_dir, "*.png")))
    template_features = prepare_template_features(templates)
    for frame_path in frame_paths:
        mask_name = generate_mask(frame_path, template_features, mask_dir, match_threshold)
        if mask_name is not None:
            inpaint_frame(frame_path, mask_name, cleaned_dir)

## Multi-threaded version
def process_all_frames_thread(frames_dir, templates, mask_dir, cleaned_dir, match_threshold=10, nTasks=1):
    """
    Processes all frames in parallel using threads.
    """
    frame_paths = sorted(glob(os.path.join(frames_dir, "*.png")))
    template_features = prepare_template_features(templates)
    
    def process_frame(frame_path):
        mask_name = generate_mask(frame_path, template_features, mask_dir, match_threshold)
        if mask_name is not None:
            inpaint_frame(frame_path, mask_name, cleaned_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=nTasks) as executor:
        futures = [executor.submit(process_frame, frame_path) for frame_path in frame_paths]
        concurrent.futures.wait(futures)

## Multi-tasking version (multiprocessing)
def process_frame_mt_worker(frame_paths, templates, mask_dir, cleaned_dir, match_threshold):
    """
    Worker function for processing a chunk of frames in a separate process.
    """
    template_features = prepare_template_features(templates)
    for frame_path in frame_paths:
        mask_name = generate_mask(frame_path, template_features, mask_dir, match_threshold)
        if mask_name is not None:
            inpaint_frame(frame_path, mask_name, cleaned_dir)

def process_all_frames_task(frames_dir, templates, mask_dir, cleaned_dir, match_threshold=10, nTasks=1):
    """
    Processes all frames in parallel using multiple processes.
    """
    frame_paths = sorted(glob(os.path.join(frames_dir, "*.png")))
    
    def split_list(lst, n):
        """Splits a list `lst` into `n` approximately equal parts."""
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    
    frame_chunks = split_list(frame_paths, nTasks)
    args_list = [(chunk, templates, mask_dir, cleaned_dir, match_threshold) for chunk in frame_chunks]

    with Pool(processes=nTasks) as pool:
        pool.starmap(process_frame_mt_worker, args_list)

# 7. Reassemble frames into a video
## Using OpenCV
def is_codec_supported(codec, test_path="test_codec.mp4", width=640, height=480, fps=25):
    """
    Checks if a given codec is supported by OpenCV on this system.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
    if not out.isOpened():
        return False
    out.release()
    os.remove(test_path)
    return True

def frames_to_video(frame_dir, output_video_path, fps=25):
    """
    Assembles PNG frames into a video file using OpenCV.
    """
    frame_paths = sorted(glob(os.path.join(frame_dir, "*.png")))
    if not frame_paths:
        return
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    if is_codec_supported('H264'):
        print("H264 compression")
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        print("H264 not available, using no compression")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

## Using ffmpeg
def frames_to_video_ffmpeg(frame_dir, output_video_path, fps=25):
    """
    Creates a video from PNG frames using ffmpeg with H.264 compression (CRF 23, preset slow).
    Raises an error if ffmpeg is not installed or no frames are found.
    """
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg is not installed or not available in PATH.")

    list_file = os.path.join(frame_dir, "ffmpeg_input.txt")
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not frame_files:
        raise FileNotFoundError("No PNG frames found in the directory.")

    with open(list_file, "w") as f:
        for frame in frame_files:
            f.write(f"file '{os.path.join(frame_dir, frame)}'\n")

    intermediate_video = os.path.join(frame_dir, "temp_output.mp4")

    subprocess.run([
        "ffmpeg", "-y", "-r", str(fps), "-f", "concat", "-safe", "0",
        "-i", list_file, "-vsync", "vfr", "-pix_fmt", "yuv420p", intermediate_video
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", intermediate_video,
        "-vcodec", "libx264", "-crf", "23", "-preset", "slow",
        output_video_path
    ], check=True)

    os.remove(list_file)
    os.remove(intermediate_video)

    print(f"Video successfully created: {output_video_path}")

# 0. Main program entry point
if __name__ == "__main__":
    video_path = "../input_video.mp4"
    logo_dir = "./in_logo"
    frame_dir = "./frames"
    mask_dir = "./masks"
    cleaned_dir = "./cleaned_frames"
    output_video = "output_video.mp4"
    match_threshold = 10
    nTasks = 5

    start_time = time.time()
    # Step 1: Extract frames from video
    print("Extracting frames...")
    #extract_frames(video_path, frame_dir)

    # Step 2: Load logo templates
    print("Loading logo templates...")
    templates = load_logo_templates(logo_dir)

    # Step 3-6: Generate masks and perform inpainting
    print("Starting mask generation and inpainting...")
    process_all_frames_thread(frame_dir, templates, mask_dir, cleaned_dir, match_threshold, nTasks)

    # Step 7: Optionally reassemble video
    print("Reassembling video...")
    frames_to_video_ffmpeg(cleaned_dir, output_video)

    end_time = time.time()
    print(f"Processing time for all frames: {end_time - start_time:.2f} seconds with {nTasks} tasks")
    print("Done.")
