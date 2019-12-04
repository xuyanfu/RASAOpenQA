import argparse
from datetime import datetime
from os.path import exists, join
import pickle
import os
import shutil


from docqa import model_dir
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from docqa.scripts.ablate_triviaqa import get_model
from docqa.text_preprocessor import WithIndicators
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion

from docqa import configurable

import  tensorflow as tf
from docqa.evaluator import EvaluatorRunner
from docqa.trainer import _build_train_ops
import numpy as np
import time
import logging
from docqa.trainer import submain
from docqa.model import Model
from docqa.model_dir import ModelDir


def save_train_start(out,data,global_step,evaluators,train_params,notes):
    """ Record the training parameters we are about to use into `out`  """

    if notes is not None:
        with open(join(out, "train_from_%d_notes.txt" % global_step), "w") as f:
            f.write(notes)

    import socket
    hostname = socket.gethostname()
    train = dict(train_params=train_params,
                 data=data,
                 start_at=global_step,
                 evaluators=evaluators,
                 date=datetime.now().strftime("%m%d-%H%M%S"),
                 host=hostname)
    with open(join(out, "train_from_%d.json" % global_step), "w") as f:
        f.write(configurable.config_to_json(train, indent=2))
    with open(join(out, "train_from_%d.pkl" % global_step), "wb") as f:
        pickle.dump(train, f)



def init(out: ModelDir, model: Model, override=False):
    """ Save our intial setup into `out` """

    for dir in [out.save_dir, out.log_dir]:
        if os.path.exists(dir):
            if len(os.listdir(dir)) > 0:
                if override:
                    print("Clearing %d files/dirs that already existed in %s" % (len(os.listdir(dir)), dir))
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                else:
                    raise ValueError()
        else:
            os.makedirs(dir)

    # JSON config just so we always have a human-readable dump of what we are working with
    with open(join(out.dir, "model.json"), "w") as f:
        f.write(configurable.config_to_json(model, indent=2))

    # Actual model saved via pickle
    with open(join(out.dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

def main():

    parser = argparse.ArgumentParser(description='Train a model on TriviaQA unfiltered')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("-t", "--n_tokens", default=400, type=int,
                        help="Paragraph size")
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with")
    parser.add_argument("-cl", "--cl", default=0, type=int, help="continue learning")
    parser.add_argument("-out", "--out", default='result/model', type=str, help="path to model")

    args = parser.parse_args()
    flag_continue = args.cl

    if flag_continue == 0:
        out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")
    if flag_continue == 1:
        out = args.out

    mode = args.mode

    model = get_model(100, 140, mode, WithIndicators())#画图
    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(args.n_tokens),ShallowOpenWebRanker(16),model.preprocessor, intern=True)
    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge", per_doc=False)]
    oversample = [1] * 4

    n_epochs = 140
    test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)#batch_size
    train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)#batch_size

    data = TriviaQaOpenDataset()

    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=1)),
        num_epochs=n_epochs, ema=0.999, max_checkpoints_to_keep=30,
        async_encoding=0, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=None, train=6000)
    )

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)
    data.preprocess(args.n_processes)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "Mode: " + args.mode + "\n" + notes

    checkpoint = None
    save_start = True

    parameter_checkpoint = None


    train_params = params
    evaluators = eval
    out = model_dir.ModelDir(out)

    if flag_continue == 0:

        print("Initializing model at: " + out.dir)
        model.init(data.get_train_corpus(), data.get_resource_loader())
        init(out, model, False)
        checkpoint = None
        save_start = True


    elif flag_continue == 1:

        if not exists(out.dir) or os.listdir(out.dir) == 0:
            print('the model dir does not exist')
            checkpoint = None
            save_start = True
        else:
            with open(join(out.dir, "model.pkl"), "rb") as f:
                model = pickle.load(f)
            latest = out.get_checkpoint(1080) # load step
            if latest is None:
                raise ValueError("No checkpoint to resume from found in " + out.save_dir)

            checkpoint = latest
            save_start = False


    if train_params.best_weights is not None:
        raise NotImplementedError

    origin_data = data
    # spec the model for the current voc/input/batching
    train = data.get_train()
    eval_datasets = data.get_eval()
    loader = data.get_resource_loader()
    evaluator_runner = EvaluatorRunner(evaluators, model)

    print("Training on %d batches" % len(train))
    print("Evaluation datasets: " + " ".join("%s (%d)" % (name, len(data)) for name, data in eval_datasets.items()))

    print("Init model...")
    model.set_inputs([train] + list(eval_datasets.values()), loader)

    print("Setting up model prediction / tf...")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    with sess.as_default():
        pred = model.get_prediction()
    evaluator_runner.set_input(pred)

    loss, summary_tensor, train_opt, global_step, _ = _build_train_ops(train_params)

    # Pre-compute tensors we need at evaluations time
    eval_tensors = []
    for ev in evaluators:
        eval_tensors.append(ev.tensors_needed(pred))

    saver = tf.train.Saver(max_to_keep=train_params.max_checkpoints_to_keep)
    summary_writer = tf.summary.FileWriter(out.log_dir)

    # Load or initialize the model parameters
    if checkpoint is not None:
        print("Restoring training from checkpoint...")
        saver.restore(sess, checkpoint)
        print("Loaded checkpoint: " + str(sess.run(global_step)))
    else:
        if parameter_checkpoint is not None:
            print("Initializing training variables...")
            vars = [x for x in tf.global_variables() if x not in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            sess.run(tf.variables_initializer(vars))
        else:
            print("Initializing parameters...")
            sess.run(tf.global_variables_initializer())

    # Make sure no bugs occur that add to the graph in the train loop, that can cause (eventuall) OOMs
    tf.get_default_graph().finalize()


    print("Start training!")

    on_step = sess.run(global_step)
    if save_start:
        summary_writer.add_graph(sess.graph, global_step=on_step)
        save_train_start(out.dir, data, on_step, evaluators, train_params, notes)

    if train_params.eval_at_zero:
        print("Running evaluation...")
        start_eval = False
        for name, data in eval_datasets.items():
            n_samples = train_params.eval_samples.get(name)
            evaluation = evaluator_runner.run_evaluators(sess, data, name, n_samples)
            for s in evaluation.to_summaries(name + "-"):
                summary_writer.add_summary(s, on_step)

    batch_time = 0
    max_f1 = 0
    for epoch in range(train_params.num_epochs):
        for batch_ix, batch in enumerate(train.get_epoch()):
            t0 = time.perf_counter()
            on_step = sess.run(global_step) + 1  # +1 because all calculations are done after step

            get_summary = on_step % train_params.log_period == 0
            encoded = model.encode(batch, True)

            if get_summary:
                summary, _, batch_loss = sess.run([summary_tensor, train_opt, loss], feed_dict=encoded)
            else:
                summary = None
                _, batch_loss = sess.run([train_opt, loss], feed_dict=encoded)

            if np.isnan(batch_loss):
                raise RuntimeError("NaN loss!")

            batch_time += time.perf_counter() - t0
            if get_summary:
                print("on epoch=%d batch=%d step=%d time=%.3f" %
                      (epoch, batch_ix + 1, on_step, batch_time))
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="time", simple_value=batch_time)]),
                                           on_step)
                summary_writer.add_summary(summary, on_step)
                batch_time = 0

        print("Checkpointing")
        saver.save(sess, join(out.save_dir, "checkpoint-" + str(110)))
        g_evaluate = tf.Graph()
        with g_evaluate.as_default():
            tmp_em, tmp_f1 = submain(out.dir)
        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            saver.save(sess, join(out.save_dir, "checkpoint-" + '1080'))

        logging.info('max_f1 {}'.format(max_f1))
        logging.info('tmp_f1 {}'.format(tmp_f1))
        logging.info('max_em {}'.format(tmp_em))
        # todo 1
        origin_data.preprocess(epoch, 1000)
        train = origin_data.get_train()

        if (epoch + 1) % 10 == 0:
            saver.save(sess, join(out.log_dir, "checkpoint-" + str(epoch)))

    sess.close()


if __name__ == "__main__":
    main()
