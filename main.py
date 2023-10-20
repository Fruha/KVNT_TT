import hydra
import contextlib
from omegaconf import DictConfig
import time
from pathlib import Path
import torch
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import glob
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.data import feature_to_text_dataset
from nemo.collections.asr.models import ASRModel, EncDecClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_segment_table,
    get_vad_stream_status,
    init_frame_vad_model,
)
from tqdm import tqdm
import os
import json
import logging
import shutil

logging.getLogger('nemo_logger').setLevel(logging.ERROR)


try:
    from torch.cuda.amp import autocast
except ImportError:
    @contextlib.contextmanager
    def autocast(enabled=None):
        yield


def extract_audio_features(cfg: DictConfig):
    file_list = []
    manifest_data = []
    manifest_filepath = str(Path(cfg.tempdir) / Path("temp_manifest_input.json"))
    out_dir = Path(cfg.tempdir) / Path("features")
    new_manifest_filepath = str(Path(cfg.tempdir) / Path("temp_manifest_input_feature.json"))

    if Path(new_manifest_filepath).is_file():
        return new_manifest_filepath

    for wav_path in glob.glob(cfg.input_data):
            item = {
                "audio_filepath": wav_path, 
                "offset": 0, 
                "duration": 10000, 
                "label": "infer", 
                "text": "-"
                }
            manifest_data.append(item)
            file_list.append(Path(item['audio_filepath']).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_filepath, manifest_data)
    torch.set_grad_enabled(False)
    if cfg.vad_model:
        vad_model = init_frame_vad_model(cfg.vad_model)
    else:
        vad_model = EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'vad_stream': False,
            'sample_rate': cfg.sample_rate,
            'manifest_filepath': manifest_filepath,
            'labels': ['infer',],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'normalize_audio_db': cfg.vad.parameters.normalize_audio_db,
        }
    )

    for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
        test_batch = [x.to(vad_model.device) for x in test_batch]
        with autocast():
            processed_signal, processed_signal_length = vad_model.preprocessor(
                input_signal=test_batch[0], length=test_batch[1],
            )
            processed_signal = processed_signal.squeeze(0)[:, :processed_signal_length]
            processed_signal = processed_signal.cpu()
            outpath = os.path.join(out_dir, file_list[i] + ".pt")
            outpath = str(Path(outpath).absolute())
            torch.save(processed_signal, outpath)
            manifest_data[i]["feature_file"] = outpath
            del test_batch

    write_manifest(new_manifest_filepath, manifest_data)
    return new_manifest_filepath

def generate_vad_frame_pred(
    vad_model: EncDecClassificationModel,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    manifest_vad_input: str,
    out_dir: str,
    use_feat: bool = False,
) -> str:
    """
    Generate VAD frame level prediction and write to out_dir
    """
    time_unit = int(window_length_in_sec / shift_length_in_sec)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    data = []
    with open(manifest_vad_input, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            file = json.loads(line)['audio_filepath'].split("/")[-1]
            data.append(file.split(".wav")[0])

    status = get_vad_stream_status(data)

    for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
        test_batch = [x.to(vad_model.device) for x in test_batch]
        with autocast():
            if use_feat:
                log_probs = vad_model(processed_signal=test_batch[0], processed_signal_length=test_batch[1])
            else:
                log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])

            probs = torch.softmax(log_probs, dim=-1)
            if len(probs.shape) == 3:
                # squeeze the batch dimension, since batch size is 1
                probs = probs.squeeze(0)  # [1,T,C] -> [T,C]
            pred = probs[:, 1]

            if window_length_in_sec == 0:
                to_save = pred
            elif status[i] == 'start':
                to_save = pred[:-trunc]
            elif status[i] == 'next':
                to_save = pred[trunc:-trunc_l]
            elif status[i] == 'end':
                to_save = pred[trunc_l:]
            else:
                to_save = pred

            to_save = to_save.cpu().tolist()
            all_len += len(to_save)

            outpath = os.path.join(out_dir, data[i] + ".frame")
            with open(outpath, "a", encoding='utf-8') as fout:
                for p in to_save:
                    fout.write(f'{p:0.4f}\n')

            del test_batch
            if status[i] == 'end' or status[i] == 'single':
                all_len = 0
    return out_dir

def run_vad_inference(cfg: DictConfig, manifest_filepath: str):

    vad_model = init_frame_vad_model(cfg.vad_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()


    test_data_config = {
        'vad_stream': True,
        'manifest_filepath': manifest_filepath,
        'labels': ['infer',],
        'num_workers': cfg.num_workers,
        'shuffle': False,
        'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
        'shift_length_in_sec': cfg.vad.parameters.shift_length_in_sec,
    }
    vad_model.setup_test_data(test_data_config=test_data_config, use_feat=True)
    pred_dir = Path(cfg.tempdir) / Path("vad_frame_pred")

    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
        shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        manifest_vad_input=manifest_filepath,
        out_dir=str(pred_dir),
        use_feat=True
    )

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    segment_dir_name = "vad_rttm"
    for key, val in cfg.vad.parameters.postprocessing.items():
        segment_dir_name = segment_dir_name + "-" + str(key) + str(val)

    segment_dir = Path(cfg.tempdir) / Path(segment_dir_name)
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = generate_vad_segment_table(
        vad_pred_dir=pred_dir,
        postprocessing_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
        out_dir=segment_dir,
        use_rttm=True,
    )

    rttm_map = {}
    for filepath in Path(segment_dir).glob("*.rttm"):
        rttm_map[filepath.stem] = str(filepath.absolute())

    manifest_data = read_manifest(manifest_filepath)
    for i in range(len(manifest_data)):
        key = Path(manifest_data[i]["audio_filepath"]).stem
        manifest_data[i]["rttm_file"] = rttm_map[key]

    new_manifest_filepath = str(Path(cfg.tempdir) / Path(f"temp_manifest_{segment_dir_name}.json"))
    write_manifest(new_manifest_filepath, manifest_data)
    return new_manifest_filepath


def init_asr_model(model_path: str) -> ASRModel:
    if model_path.endswith('.nemo'):
        asr_model = ASRModel.restore_from(restore_path=model_path)
    elif model_path.endswith('.ckpt'):
        asr_model = ASRModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        asr_model = ASRModel.from_pretrained(model_name=model_path)
    return asr_model

def run_asr_inference(cfg: DictConfig, manifest_filepath: str) -> str:
    asr_model = init_asr_model(cfg.asr_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device)
    asr_model.eval()

    # Setup decoding strategy
    decode_function = None
    decoder_type = cfg.get("decoder_type", None)
    if not hasattr(asr_model, 'change_decoding_strategy'):
        raise ValueError(f"ASR model {cfg.asr_model} does not support decoding strategy.")
    if decoder_type is not None:  # Hybrid model
        if decoder_type == 'rnnt':
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding, decoder_type=decoder_type)
            decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
        elif decoder_type == 'ctc':
            asr_model.change_decoding_strategy(cfg.ctc_decoding, decoder_type=decoder_type)
            decode_function = asr_model.decoding.ctc_decoder_predictions_tensor
        else:
            raise ValueError(
                f"Unknown decoder type for hybrid model: {decoder_type}, supported types: ['rnnt', 'ctc']"
            )
    elif hasattr(asr_model, 'joint'):  # RNNT model
        cfg.rnnt_decoding.fused_batch_size = -1
        cfg.rnnt_decoding.compute_langs = cfg.compute_langs
        asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
    else:
        asr_model.change_decoding_strategy(cfg.ctc_decoding)
        decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    data_config = {
        "manifest_filepath": manifest_filepath,
        "normalize": cfg.normalize,
        "normalize_type": cfg.normalize_type,
        "use_rttm": cfg.vad.use_rttm,
        "rttm_mode": cfg.rttm_mode,
        "feat_mask_val": cfg.feat_mask_val,
        "frame_unit_time_secs": cfg.frame_unit_time_secs,
    }
    if hasattr(asr_model, "tokenizer"):
        dataset = feature_to_text_dataset.get_bpe_dataset(config=data_config, tokenizer=asr_model.tokenizer)
    else:
        data_config["labels"] = asr_model.decoder.vocabulary
        dataset = feature_to_text_dataset.get_char_dataset(config=data_config)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset._collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )

    hypotheses = []
    all_hypotheses = []
    t0 = time.time()
    with autocast():
        with torch.no_grad():
            for test_batch in tqdm(dataloader, desc="Transcribing"):
                outputs = asr_model.forward(
                    processed_signal=test_batch[0].to(device),
                    processed_signal_length=test_batch[1].to(device),
                )

                logits, logits_len = outputs[0], outputs[1]

                current_hypotheses, all_hyp = decode_function(logits, logits_len, return_hypotheses=False,)
                if isinstance(current_hypotheses, tuple) and len(current_hypotheses) == 2:
                    current_hypotheses = current_hypotheses[0]

                hypotheses += current_hypotheses
                if all_hyp is not None:
                    all_hypotheses += all_hyp
                else:
                    all_hypotheses += current_hypotheses

                del logits
                del test_batch

    manifest_data = read_manifest(manifest_filepath) 

    for i, item in enumerate(manifest_data):
        print(f"filepath: {item['audio_filepath']}")
        print(f"Transcribition: {hypotheses[i]} \n")

    return cfg.output_filename

@hydra.main(version_base=None, config_path="", config_name="params")
def main(cfg : DictConfig) -> None:
    if cfg.clear_cache:
        shutil.rmtree(cfg.tempdir,ignore_errors=True)
    cfg.ctc_decoding = CTCDecodingConfig()
    manifest_filepath = extract_audio_features(cfg)
    manifest_filepath = run_vad_inference(cfg, manifest_filepath)
    run_asr_inference(cfg, manifest_filepath)
    if cfg.clear_cache:
        shutil.rmtree(cfg.tempdir,ignore_errors=True)

if __name__ == '__main__':
    main()