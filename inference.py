import argparse
import cv2
import librosa
import librosa.filters
import numpy as np
import os
import platform
import subprocess
import torch
from glob import glob
from models.Architecture import Wav2Lip
from os import listdir, path
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
import face_detection

# Argument Parsing
parser = argparse.ArgumentParser(description='Inference using Wav2Lip model')
parser.add_argument('--checkpoint', type=str, help='Path of weights', default='checkpoints/wav2lip_gan.pth', required=False)
parser.add_argument('--image', type=str, help='Image Path', required=True)
parser.add_argument('--audio', type=str, help='Audio Path', required=True)
parser.add_argument('--output', type=str, help='Video path', default='output/result.mp4')
parser.add_argument('--fps', type=float, help='default: 25', default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding [top, bottom, left, right]')
parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. best output 720p')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box for the face.')
args = parser.parse_args()

# Global Variables  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


# Function Definitions

def face_detect(images):
    """
    Detect faces in images.
    """
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    batch_size = args.face_det_batch_size
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, mels):
    """
    Data generator for model inference.
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        face_det_results = face_detect([frames[0]]) 
    else:
        print('Using the specified bounding box instead of face detection.')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        
    idx = 0
    frame_to_save = frames[idx].copy()
    face, coords = face_det_results[idx].copy()
    batch_size = args.wav2lip_batch_size
    face = cv2.resize(face, (96, 96))
    
    num_batches = int(np.ceil(len(mels) / batch_size))
    for i in range(num_batches):
        if i < num_batches - 1:
            batch_mels = mels[i * batch_size: (i + 1) * batch_size]
            batch_size_actual = batch_size
        else:
            batch_mels = mels[i * batch_size:]
            batch_size_actual = len(batch_mels)

        img_batch.extend([face] * batch_size_actual)
        frame_batch.extend([frame_to_save] * batch_size_actual)
        coords_batch.extend([coords] * batch_size_actual)

        img_masked = np.array(img_batch)
        img_masked[:, 96 // 2:] = 0

        img_batch = np.concatenate((img_masked, np.array(img_batch)), axis=3) / 255.
        mel_batch = np.reshape(np.array(batch_mels), [len(batch_mels), batch_mels[0].shape[0], batch_mels[0].shape[1], 1])


        yield img_batch, mel_batch, frame_batch, coords_batch
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

def load_checkpoint(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint
        
def load_model(path):
    """
    Load the Wav2Lip model.
    """
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = load_checkpoint(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def melspectrogram(wav):
    """
    Compute the mel spectrogram from wav.
    """
    y = signal.lfilter([1, -0.97], [1], wav)
    D = librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)
    _mel_basis = librosa.filters.mel(16000, 800, n_mels=80,fmin=55, fmax=7600)

    min_level = np.exp(-100 / 20 * np.log(10))
    S = (20 * np.log10(np.maximum(min_level, np.dot(_mel_basis, np.abs(D))))) - 20
    _normalize = np.clip((2 * 4.) * ((S + 100) / (100)) - 4., -4., 4.)
    
    return _normalize

def main():
    """
    Main function to run inference.
    """
    if not os.path.isfile(args.image):
        raise ValueError('--face argument must be a valid path to image file')

    elif args.image.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.image)]
        fps = args.fps

    else:
        raise ValueError('--face argument must be a image file')

    print ("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = librosa.core.load(args.audio, sr=16000)[0]
    mel = melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + 16 > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - 16:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + 16])
        i += 1

    print("####Length of mel chunks: {}#####".format(len(mel_chunks)))

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.output)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
