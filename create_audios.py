import subprocess
import pysrt
from pathlib import Path
from pydub import AudioSegment
import pandas as pd

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np

def get_audio_from_video():
    input_path_name = "Input"
    path_list = Path(input_path_name).iterdir()
    output_path_name = "Output/"
    output_format = "wav"
    for path_input in path_list:
        path_output = output_path_name + str(path_input)[(len(input_path_name) + 1):-4] + "." + output_format
        subprocess.call(
            ['ffmpeg', '-i', path_input, '-f', output_format, '-ac', '1', '-rf64', 'auto', '-vn',path_output])
        #  ['ffmpeg', '-i', path_input, '-q:a', '0', '-map', 'a', path_output])
    return 0


def slice_audio(full_file_name, format, window):
    file_name = full_file_name[7:-4]
    if (format is not "wav"):
        audio = AudioSegment.from_file(full_file_name, format)
    else:
        audio = AudioSegment.from_wav(full_file_name)
    audio_duration = (len(audio) / (1000))  # In seconds
    subprocess.call(["mkdir", "Slices/" + file_name])
    subprocess.call(["mkdir", "Slices/" + file_name + '/talking'])
    subprocess.call(["mkdir", "Slices/" + file_name + '/not_talking'])
    subs_name = "{0}/{1}.srt".format("Subtitles", file_name)
    subs = pysrt.open(subs_name)
    subs_index = 0
    current_end = 0



    for slice_num in range(round(audio_duration / window)):
        talking = False
        current_end += window
        subs_start_in_seconds = subs[subs_index].start.hours*3600+ subs[subs_index].start.minutes*60 + subs[subs_index].start.seconds
        subs_end_in_seconds = subs[subs_index].end.hours*3600+ subs[subs_index].end.minutes*60 + subs[subs_index].end.seconds
        if subs_start_in_seconds < current_end:
            talking = True

        sliced = audio[slice_num * (window * 1000): (slice_num + 1) * (window * 1000)]
        saved_path = "Slices/{0}/{1}/{2}_{3}.{4}"
        sub_dir = 'not_talking'
        if talking:
            sub_dir = 'talking'

        saved_path = saved_path.format(file_name, sub_dir, file_name, slice_num, format)
        sliced.export(saved_path, format, tags = {'talking': talking})
        while subs_end_in_seconds < current_end:
            if subs_index == len(subs) - 1:
                break
            subs_index += 1
            subs_end_in_seconds = subs[subs_index].end.hours * 3600 + subs[subs_index].end.minutes * 60 + subs[
                subs_index].end.seconds


def create_vector_from_wav(file_name, talking):
    (rate, sig) = wav.read(file_name)
    mfcc_feat = mfcc(sig, rate, nfft=1200)
    return pd.DataFrame(mfcc_feat)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # # print(d_mfcc_feat)
    # fbank_feat = logfbank(sig, rate, nfft=1200)

    # print(fbank_feat[1:3, :])


def run():
    get_audio_from_video()
    path_list = Path("Output").iterdir()
    for audio_path in path_list:
        try:
            slice_audio(str(audio_path), "wav", 1)
        except FileNotFoundError as error:
            print("File {0} not found.".format(error.filename))
    # path_list = Path("Slices").iterdir()
    # talking_data = pd.DataFrame()
    # not_talking_data = pd.DataFrame()
    # for film_path in path_list:
    #     talking_path = Path("{0}/talking".format(str(film_path))).iterdir()
    #     not_talking_path = Path("{0}/not_talking".format(str(film_path))).iterdir()
    #     for audio_talking in talking_path:
    #         talking_data = talking_data.append(create_vector_from_wav(audio_talking, 1) )
    #     for audio_not_talking in not_talking_path:
    #         not_talking_data = not_talking_data.append(create_vector_from_wav(audio_not_talking, 0))
    # talking_data.to_csv("talking_data.csv", header=None, index=False)
    # not_talking_data.to_csv("not_talking_data.csv", header=None, index=False)
    return 0


