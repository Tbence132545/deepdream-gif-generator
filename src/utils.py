from PIL import Image
import os

def save_gif(frames_dir, output_file, num_frames, frame_delay):
    frames = [Image.open(os.path.join(frames_dir, f"frame_{i:04d}.png")) for i in range(num_frames)]
    frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=frame_delay, loop=0)
    print(f"Animation saved to {output_file}")
