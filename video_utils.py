import os
import cv2
import subprocess
import time
import numpy as np
from pathlib import Path

def reencode_video_to_16fps(input_video_path, num_frames, target_width=None, target_height=None):
    """
    Re-encodes the input video to 16 FPS and trims it to match the desired frame count.
    Also handles resizing to match target dimensions if provided.
    
    Args:
        input_video_path: Path to the input video
        num_frames: Number of frames requested by the user
        target_width: Target width for the output video (optional)
        target_height: Target height for the output video (optional)
        
    Returns:
        Path to the re-encoded video
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[CMD] Could not open video {input_video_path}")
        return input_video_path
    
    # Get input video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If target dimensions aren't specified, use input dimensions
    if target_width is None:
        target_width = input_width
    if target_height is None:
        target_height = input_height
        
    print(f"[CMD] Input video: {total_frames} frames at {input_fps} FPS, dimensions: {input_width}x{input_height}")
    print(f"[CMD] Target: {num_frames} frames at 16 FPS, dimensions: {target_width}x{target_height}")
    
    # Determine if reencoding is needed
    needs_reencoding = False
    
    if abs(input_fps - 16) > 0.01:
        print(f"[CMD] Input video needs re-encoding: FPS is {input_fps} instead of 16")
        needs_reencoding = True
    
    if total_frames > (num_frames + 5) or total_frames < num_frames:
        print(f"[CMD] Input video needs re-encoding: frame count mismatch ({total_frames} vs {num_frames})")
        needs_reencoding = True
    
    if input_width != target_width or input_height != target_height:
        print(f"[CMD] Input video needs re-encoding: dimension mismatch ({input_width}x{input_height} vs {target_width}x{target_height})")
        needs_reencoding = True
    
    if not needs_reencoding:
        cap.release()
        print(f"[CMD] Video already meets requirements, no re-encoding needed")
        return input_video_path
    
    # Create output directory
    output_folder = "auto_pre_processed_videos"
    os.makedirs(output_folder, exist_ok=True)
    
    timestamp = int(time.time())
    reencoded_video = os.path.join(output_folder, f"reencoded_{timestamp}_{Path(input_video_path).name}")
    frames_dir = os.path.join(output_folder, f"temp_frames_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # Step 1: Extract exactly num_frames frames, with proper resizing and aspect ratio
        frames = []
        
        # Calculate input aspect ratio
        input_aspect = input_width / input_height
        target_aspect = target_width / target_height
        
        # Determine how many frames to extract from input
        frames_to_extract = min(num_frames, total_frames)
        
        # For any missing frames, we'll duplicate the first frame
        missing_frames = max(0, num_frames - total_frames)
        if missing_frames > 0:
            print(f"[CMD] Input video has fewer frames than needed. Will duplicate first frame {missing_frames} times.")
        
        # Read the first frame (which might need to be duplicated)
        success, first_frame = cap.read()
        if not success:
            cap.release()
            print(f"[CMD] Could not read first frame from video")
            return input_video_path
            
        # Process the first frame (resize/crop)
        if abs(input_aspect - target_aspect) < 0.01:
            # Simple resize if aspect ratios are close enough
            first_frame_processed = cv2.resize(first_frame, (target_width, target_height))
        else:
            # Need to crop and scale to maintain aspect ratio
            if input_aspect > target_aspect:
                # Input is wider than target - crop width
                new_width = int(input_height * target_aspect)
                crop_x = int((input_width - new_width) / 2)
                cropped = first_frame[:, crop_x:crop_x+new_width]
                first_frame_processed = cv2.resize(cropped, (target_width, target_height))
            else:
                # Input is taller than target - crop height
                new_height = int(input_width / target_aspect)
                crop_y = int((input_height - new_height) / 2)
                cropped = first_frame[crop_y:crop_y+new_height, :]
                first_frame_processed = cv2.resize(cropped, (target_width, target_height))
        
        # Add duplicated first frames if needed
        for i in range(missing_frames):
            frames.append(first_frame_processed.copy())
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), first_frame_processed)
            
        # Process the remaining frames
        frame_count = missing_frames
        frames.append(first_frame_processed)  # Add the actual first frame
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:06d}.png"), first_frame_processed)
        frame_count += 1
        
        # If we need more frames, continue reading from input
        if frame_count < num_frames:
            # Calculate frame step to evenly distribute frames if input has more frames than needed
            if total_frames > 1 and frames_to_extract > 1:
                frame_step = (total_frames - 1) / (frames_to_extract - 1)
            else:
                frame_step = 1
                
            current_pos = 0
            
            while frame_count < num_frames and current_pos < total_frames - 1:
                # Calculate next frame to extract
                next_pos = int(min(current_pos + frame_step, total_frames - 1))
                if next_pos <= current_pos:
                    next_pos = current_pos + 1
                
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame (resize/crop)
                if abs(input_aspect - target_aspect) < 0.01:
                    # Simple resize
                    frame_processed = cv2.resize(frame, (target_width, target_height))
                else:
                    # Crop and scale
                    if input_aspect > target_aspect:
                        # Input is wider than target - crop width
                        new_width = int(input_height * target_aspect)
                        crop_x = int((input_width - new_width) / 2)
                        cropped = frame[:, crop_x:crop_x+new_width]
                        frame_processed = cv2.resize(cropped, (target_width, target_height))
                    else:
                        # Input is taller than target - crop height
                        new_height = int(input_width / target_aspect)
                        crop_y = int((input_height - new_height) / 2)
                        cropped = frame[crop_y:crop_y+new_height, :]
                        frame_processed = cv2.resize(cropped, (target_width, target_height))
                
                frames.append(frame_processed)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:06d}.png"), frame_processed)
                
                # Update positions
                current_pos = next_pos
                frame_count += 1
        
        cap.release()
        
        # Ensure we have exactly num_frames
        if len(frames) != num_frames:
            print(f"[CMD] Warning: Extracted {len(frames)} frames, but target is {num_frames}")
            # If we have too few frames, duplicate the last frame
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
                cv2.imwrite(os.path.join(frames_dir, f"frame_{len(frames)-1:06d}.png"), frames[-1])
            # If we have too many frames, truncate
            if len(frames) > num_frames:
                frames = frames[:num_frames]
        
        print(f"[CMD] Successfully extracted and processed {len(frames)} frames")
        
        # Step 2: Encode the frames to a video
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', '16',
            '-i', os.path.join(frames_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-level', '3.1',
            '-preset', 'veryslow',
            '-crf', '12',
            '-pix_fmt', 'yuv420p',
            '-x264-params', 'ref=4:cabac=1',
            '-an',
            reencoded_video
        ]
        
        print(f"[CMD] Encoding frames to video at 16 FPS...")
        subprocess.run(' '.join(ffmpeg_cmd), shell=True, check=True)
        
        # Clean up temporary frames
        try:
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
            os.rmdir(frames_dir)
        except Exception as e:
            print(f"[CMD] Warning: Could not clean up temp directory: {e}")
        
        # Verify the frame count
        verify_cap = cv2.VideoCapture(reencoded_video)
        if verify_cap.isOpened():
            output_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = verify_cap.get(cv2.CAP_PROP_FPS)
            verify_cap.release()
            print(f"[CMD] Re-encoded video has {output_frames} frames at {output_fps} FPS (target: {num_frames} frames at 16 FPS)")
            
            if output_frames != num_frames:
                print(f"[CMD] Warning: Output video has {output_frames} frames, but target is {num_frames}")
            
            if abs(output_fps - 16) > 0.01:
                print(f"[CMD] Warning: Output FPS is {output_fps}, not 16")
        
        print(f"[CMD] Video re-encoded successfully: {reencoded_video}")
        return reencoded_video
    
    except Exception as e:
        cap.release()
        print(f"[CMD] Error during video re-encoding: {e}")
        
        # Clean up temporary frames if they exist
        try:
            if os.path.exists(frames_dir):
                for f in os.listdir(frames_dir):
                    os.remove(os.path.join(frames_dir, f))
                os.rmdir(frames_dir)
        except:
            pass
            
        return input_video_path

def clean_temp_videos():
    """Clean up temporary videos that were created during re-encoding"""
    # Clean both the temp_videos and auto_pre_processed_videos folders
    folders_to_clean = ["temp_videos", "auto_pre_processed_videos"]
    
    for folder in folders_to_clean:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                for file in os.listdir(folder):
                    if file.startswith("reencoded_"):
                        file_path = os.path.join(folder, file)
                        try:
                            os.remove(file_path)
                            print(f"[CMD] Removed temporary video: {file_path}")
                        except Exception as e:
                            print(f"[CMD] Error removing temporary video {file_path}: {e}")
                    if file.startswith("temp_frames_"):
                        dir_path = os.path.join(folder, file)
                        try:
                            if os.path.isdir(dir_path):
                                for f in os.listdir(dir_path):
                                    os.remove(os.path.join(dir_path, f))
                                os.rmdir(dir_path)
                                print(f"[CMD] Removed temporary frames directory: {dir_path}")
                        except Exception as e:
                            print(f"[CMD] Error removing temporary frames directory {dir_path}: {e}")
            except Exception as e:
                print(f"[CMD] Error cleaning videos in {folder}: {e}") 