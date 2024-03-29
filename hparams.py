#import tensorflow as tf
#from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    #hparams = tf.contrib.training.HParams(
    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 500,
        "iters_per_checkpoint": 200,
        "seed": 1234,
        #"dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "warm_start": False,
        "checkpoint_path": None, #"tacotron2_statedict.pt",
        "ignore_layers": [], #['embedding.weight'],
        "n_gpus": 1,
        "rank": 0,
        "group_name": "isaacs_group",

        ################################
        # Data Parameters             #
        ################################
        ##"load_mel_from_disk": False,
        "output_directory": "checkpoints_sanderson",
        "ppg_dir": "../sanderson_ppg",
        "mel_dir": "../sanderson_melspec",
        "training_file_list": "filelists/sanderson_train_files.txt",
        "validation_file_list": "filelists/sanderson_val_files.txt",
        "log_directory": "log",

        #"output_directory": "checkpoints_sanderson",
        #"ppg_dir": "../antonimuthu_ppg",
        #"mel_dir": "../sanderson_melspec",
        #"training_file_list": "filelists/antonimuthu_train_files.txt",
        #"validation_file_list": "filelists/antonimuthu_val_files.txt",
        #"log_directory": "log",
        
        ##"text_cleaners": ['english_cleaners'],


        ################################
        # Audio Parameters             #
        ################################
        #"max_wav_value": 32768.0,
        #"sampling_rate": 22050,
        #"filter_length": 1024,
        #"hop_length": 256,
        #"win_length": 1024,
        "n_mel_channels": 80,
        #"mel_fmin": 0.0,
        #"mel_fmax": 8000.0,

        ################################
        # Model Parameters             #
        ################################
        #"n_symbols": len(symbols),
        "ppg_n_phonemes": 40,
        "ppg_embedding_dim": 256,
        
        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 256,

        # Decoder parameters
        "n_frames_per_step": 1,  # currently only 1 is supported
        "decoder_rnn_dim": 400,
        "prenet_dim": 256,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 400,
        "attention_dim": 128,
        "attention_window_size": 10,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,
        
        # Loss function parameters
        "mel_weight": 1.0,
        "gate_weight": 0.005,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 4,
        "mask_padding": True  # set model's padded outputs to padded values
    #)
    }

    #if hparams_string:
    #    tf.logging.info('Parsing command line hparams: %s', hparams_string)
    #    hparams.parse(hparams_string)

    #if verbose:
    #    tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
