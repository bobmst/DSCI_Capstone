# %%
import pandas as pd
import numpy as np
import pickle

import os
from pprint import pprint
import re
from collections import defaultdict

from sklearn import preprocessing
import librosa
import soundfile as sf
import os
import re

from util import *
from tqdm import tqdm

tqdm.pandas()
import importlib
import logging
import yaml
from main import *

import gradio as gr


# %%


# %%


# %%
def process_gradio_audio(audio_input, target_sr):
    origin_sr, origin_signal = audio_input
    if len(origin_signal.shape) == 2:
        origin_signal = origin_signal.T[0]
    origin_signal = origin_signal.astype(np.float32)
    signal = librosa.resample(origin_signal, orig_sr=origin_sr, target_sr=target_sr)
    return signal


def main():
    # ===Setting for all parameters===
    sample_rate = 22050
    segment_length = 5
    n_fft = 2048
    n_mels = 128
    n_mfcc = 17
    params_path = "params.yml"
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    algo = params["algorithm"]
    batch_size = int(algo["params"]["batch_size"])
    n_epochs = 50
    patience = int(algo["params"]["patience"])
    learning_rate = float(algo["params"]["learning_rate"])

    model_dir = "./model/auth/app"
    model_save_path = os.path.join(model_dir, "demo.pkl")
    demo_model_path = "./model/auth/demo"

    device = get_cuda_device()

    # CNN
    PATH = "./model/basecnn200"
    model_set = torch.load(f"{PATH}/model.pt")
    clf_pre = model_set["model"]

    DATA_ADDRESS = "./data"
    NON_SPEAKER_DIR = os.path.join(DATA_ADDRESS, "preprocessed", "test")
    X_false, _ = load_data(
        dir_feature=NON_SPEAKER_DIR,
        file_prefix="source_mfcc_len5_fft2048_mels128_mfcc17_",
        dir_df_index=os.path.join(DATA_ADDRESS, "df_index_source_test.pkl"),
        n_interval=500,
        flatten=False,
    )

    # ===UI functions===
    def train_model_pipeline(audio_input, denoise, model_name):
        output = "Train Pipline Starting\nLoading..."
        yield output
        # Use the gr.Progress() to show the progress of training
        speaker_auth_model = auth_model(
            clf=clf_pre, X_false=X_false, batch_size=batch_size
        )
        output += "\nBase model loaded"
        yield output
        signal = process_gradio_audio(audio_input, sample_rate)
        output += "\nInput signal loaded"
        yield output

        speaker_audio_loader = audio_loader(ls_raw_signal=[signal], sr=sample_rate)
        speaker_audio_loader.process_raw(
            segment_length=segment_length,
            denoise=denoise,
            n_fft=n_fft,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )
        output += "\nFinished Preprocessing\nModel Training..."
        yield output

        speaker_auth_model.train(
            train_audio_loader=speaker_audio_loader,
            n_epochs=n_epochs,
            patience=patience,
            model_save_path=model_save_path,
            learning_rate=learning_rate,
        )
        output += "\nFinished Training"
        yield output

        model_dir = os.path.join(demo_model_path, model_name)
        save_pickle(model_dir, speaker_auth_model)
        print("model saved to:", model_dir)
        output += f"\nmodel saved to:{model_dir}"
        yield output
        output += "\nTraining completed successfully!"
        yield output

    def fit_model_pipeline(audio_input, denoise, model_name, accept_threshold):
        # signal,_ = read_signal(audio_input,sample_rate)
        model_dir = os.path.join(demo_model_path, model_name)
        speaker_auth_model = load_pickle(model_dir)
        print("Fine tuned model loaded")
        signal = process_gradio_audio(audio_input, sample_rate)
        print("Input signal loaded")

        speaker_audio_loader = audio_loader(ls_raw_signal=[signal], sr=sample_rate)
        speaker_audio_loader.process_raw(
            segment_length=segment_length,
            denoise=denoise,
            n_fft=n_fft,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )

        pred_list, prob_list = speaker_auth_model.predict(
            speaker_audio_loader=speaker_audio_loader,
            batch_size=batch_size,
            accept_threshold=accept_threshold,
        )
        avg_prob = np.average(prob_list[0])
        if avg_prob >= accept_threshold:
            pred = "Authorized!😁"
        else:
            pred = "Denied!😠"
        prob = "{:.4%}".format(avg_prob)
        return pred, prob

    # ===UI===
    with gr.Blocks() as demo:
        with gr.Tab("Enrollment"):
            gr.Markdown(
                "Please record or upload your enrollment audio down below, then click the `train` button."
            )
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="Audio Input for authentication")
                    denoise = gr.Checkbox(label="Denoise", value=True)
                with gr.Column():
                    train_button = gr.Button("Train")
                    train_output = gr.Textbox(label="Training Status")
            gr.Markdown("Fancy UI for nerd ↓")
            with gr.Row():
                with gr.Group():
                    # audio_input_dir = gr.Textbox(label="Audio File Directory (If you have multiple files)")
                    model_name = gr.Textbox(label="Model Name", value="demo.pkl")
                    # gr.Markdown("blablabla blablabla")

            train_button.click(
                train_model_pipeline,
                inputs=[audio_input, denoise, model_name],
                outputs=train_output,
            )

        with gr.Tab("Authentication"):
            with gr.Row():
                with gr.Column():
                    test_audio_input = gr.Audio(label="Audio Input for authentication")
                    # test_audio_input = gr.Textbox(label="Audio File Directory")
                    test_denoise = gr.Checkbox(label="Denoise", value=True)
                    threshold = gr.Number(label="Accpetance threshold", value=0.85)
                    evaluate_button = gr.Button("Auth")
                with gr.Column():
                    fit_result = gr.Textbox(label="Verification successful?", value="?")
                    confidence = gr.Textbox(
                        label="Voice verification pass rate", value=0
                    )
            gr.Markdown("---")
            gr.Markdown("Other stuff may not needed")
            with gr.Row():
                test_model_dir = gr.Textbox(
                    label="Saved Model Directory", value="demo.pkl"
                )

            evaluate_button.click(
                fit_model_pipeline,
                inputs=[test_audio_input, test_denoise, test_model_dir, threshold],
                outputs=[fit_result, confidence],
            )

        # with gr.Tab("Status"):
        #     gr.Markdown("Training status")
        #     with gr.Row():
        #         train_accuracy = gr.Textbox(label="Training Accuracy",value=np.nan)
        #         train_eer = gr.Textbox(label="Training EER",value=np.nan)
        #         train_data_length = gr.Textbox(label="Length of Training Data")
        #         test_data_length = gr.Textbox(label="Length of Test Data")
        #     gr.Markdown("Training status")
        #     with gr.Row():
        #         gr.Text("Train")
        #     with gr.Column():
        #         refresh_button = gr.Button("Refresh")
        #     refresh_button.click(foo,outputs=[train_accuracy,train_eer])

    demo.launch()


main()
