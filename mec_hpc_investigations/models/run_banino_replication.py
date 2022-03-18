import tensorflow as tf
import os
from mec_hpc_investigations.models.utils import configure_options, configure_model
from mec_hpc_investigations.models.trainer import Trainer
from mec_hpc_investigations.core.default_dirs import BANINO_REP_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None, required=True)
    parser.add_argument("--save_dir", type=str, default=BANINO_REP_DIR)
    parser.add_argument("--rnn_type", type=str, default="BaninoRNN")
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--arena_size", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=100)
    parser.add_argument("--place_cell_rf", type=float, default=0.01)
    parser.add_argument("--Np", type=int, default=256)
    parser.add_argument("--Ng", type=int, default=256)
    parser.add_argument("--Nhdc", type=int, default=12)
    parser.add_argument("--hdc_concentration", type=int, default=20)
    parser.add_argument("--exclude_hdc_loss", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optimizer_class", type=str, default="rmsprop")
    parser.add_argument("--banino_rnn_type", type=str, default="lstm")
    parser.add_argument("--banino_dropout_rate", type=float, default=0.5)
    parser.add_argument("--banino_rnn_nunits", type=int, default=128)
    parser.add_argument("--clipvalue", type=float, default=1e-5)
    parser.add_argument("--exclude_clip", type=bool, default=False)
    parser.add_argument("--run_ID", type=str, default=None)
    ARGS = parser.parse_args()

    # If GPUs available, select which to train on
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=ARGS.gpu

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    Nhdc = ARGS.Nhdc
    if ARGS.exclude_hdc_loss:
        Nhdc = None

    clipvalue = ARGS.clipvalue
    if ARGS.exclude_clip:
        clipvalue = None

    # Training options and hyperparameters
    options = configure_options(save_dir=ARGS.save_dir,
                                rnn_type=ARGS.rnn_type,
                                activation=ARGS.activation,
                                arena_size=ARGS.arena_size,
                                learning_rate=ARGS.learning_rate,
                                weight_decay=ARGS.weight_decay,
                                optimizer_class=ARGS.optimizer_class,
                                n_epochs=ARGS.n_epochs,
                                n_steps=ARGS.n_steps,
                                batch_size=ARGS.batch_size,
                                sequence_length=ARGS.sequence_length,
                                place_cell_rf=ARGS.place_cell_rf,
                                Np=ARGS.Np,
                                Ng=ARGS.Ng,
                                Nhdc=Nhdc,
                                hdc_concentration=ARGS.hdc_concentration,
                                banino_place_cell=True,
                                DoG=False,
                                banino_rnn_type=ARGS.banino_rnn_type,
                                banino_dropout_rate=ARGS.banino_dropout_rate,
                                banino_rnn_nunits=ARGS.banino_rnn_nunits,
                                clipvalue=clipvalue,
                                run_ID=ARGS.run_ID)

    model = configure_model(options, rnn_type=ARGS.rnn_type)
    trainer = Trainer(options, model)
    trainer.train(n_epochs=options.n_epochs,
                  n_steps=options.n_steps,
                  save=True)

