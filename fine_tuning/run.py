import os
import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import time
import sys
import math
import numpy as np
import functools
import re
import logging
import glob

import paddle
import paddle.fluid as fluid
from as_models import *

from fine_tuning.args import args
from as_data_reader.data_path import global_data_path
from as_data_reader.readers import ReaderConfig

from as_utils.logger import LoggerText
from as_utils.tools import AverageMeter
from as_models import ResNet50

logging.basicConfig(level=logging.INFO)
logging_logger = logging.getLogger(__name__)
if args.seed is not None:
    np.random.seed(args.seed)

print(os.environ.get('LD_LIBRARY_PATH', None))
print(os.environ.get('PATH', None))

def get_model_id():
    prefix = ''
    if args.prefix is not None:
        prefix = args.prefix + '-'  # for some notes.

    model_id = prefix + args.dataset + \
               args.model + \
               '-epo_' + str(args.num_epoch) + \
               '-b_' + str(args.batch_size) + \
               '-wd_' + str(args.wd_rate) + \
               '-fcreinit_' + str(args.fc_reinit)
    return model_id


def main():
    directory = os.path.join(f'{args.outdir}', get_model_id())
    """
    if os.path.exists(os.path.join(directory, f'snapshot')):
        print(f'{directory} exists.')
        print('No need to do the retraining.')
        return
    """

    dataset = args.dataset
    image_shape = [3, 224, 224]
    pretrained_model = args.pretrained_model
    class_map_path = f'{global_data_path}/{dataset}/readable_label.txt'


    if os.path.exists(class_map_path):
        logging_logger.info("The map of readable label and numerical label has been found!")
        with open(class_map_path) as f:
            label_dict = {}
            strinfo = re.compile(r"\d+ ")
            for item in f.readlines():
                key = int(item.split(" ")[0])
                value = [
                    strinfo.sub("", l).replace("\n", "")
                    for l in item.split(", ")
                ]
                label_dict[key] = value[0]

    assert os.path.isdir(pretrained_model), "please load right pretrained model path for infer"

    # data reader
    batch_size = args.batch_size
    reader_config = ReaderConfig(f'{os.path.join(global_data_path, dataset)}', is_test=False)
    reader = reader_config.get_reader()
    reader = paddle.reader.buffered(reader, args.batch_size * 4)
    train_reader = paddle.batch(
        paddle.reader.shuffle(reader, buf_size=batch_size),
        batch_size,
        drop_last=True)

    test_batch_size = args.test_batch_size
    reader_config = ReaderConfig(f'{os.path.join(global_data_path, dataset)}', is_test=True)
    reader = reader_config.get_reader()
    reader = paddle.reader.buffered(reader, args.test_batch_size*4)
    test_reader = paddle.batch(reader, test_batch_size)

    fine_tuning_program = fluid.Program()
    with fluid.program_guard(fine_tuning_program):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', shape=[None] + image_shape, dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            model = ResNet50()
            logits = model.net(input=image, class_dim=reader_config.num_classes)
            if isinstance(logits, tuple):
                logits = logits[0]
            out = fluid.layers.softmax(logits)
            cost = fluid.layers.mean(fluid.layers.cross_entropy(out, label))
            accuracy = fluid.layers.accuracy(input=out, label=label)

            # fine_tuning_program = fine_tuning_program.clone(for_test=False)
            test_program = fine_tuning_program.clone(for_test=True)

            # optimizer, with piecewise_decay learning rate.
            total_steps = len(reader_config.image_paths) * args.num_epoch // batch_size
            boundaries = [int(total_steps * 2 / 3)]
            print('\ttotal learning steps:', total_steps)
            print('\tlr decays at:', boundaries)
            values = [0.01, 0.001]

            step_each_epoch = np.ceil(len(reader_config.image_paths) / batch_size)
            #print(step_each_epoch)
            optimizer = fluid.optimizer.Momentum(
                #learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
                learning_rate=fluid.layers.cosine_decay(learning_rate=args.lr_init, step_each_epoch=step_each_epoch, epochs=args.num_epoch),
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(args.wd_rate)
            )
            cur_lr = optimizer._global_learning_rate()

            trainable_params = []
            for var in fine_tuning_program.list_vars():
                if hasattr(var, 'trainable') and var.trainable:
                    trainable_params.append(var)

            optimizer.minimize(cost)

    # data reader
    feed_order = ['image', 'label']

    # executor (session)
    place = fluid.CUDAPlace(args.use_cuda) if args.use_cuda >= 0 else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def _predicate(var):
        # not loading the fc layers.
        if 'fc' in var.name:
            # print('not loading', var.name)
            return False

        # load only the existed and trainable parameters
        return os.path.exists(os.path.join(pretrained_model, var.name)) and var.trainable

    fluid.io.load_vars(exe, pretrained_model, predicate=_predicate, main_program=fine_tuning_program)

    feed_var_list_loop = [
        fine_tuning_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_training = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place, program=fine_tuning_program)
    feeder_test = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place, program=test_program)

    train_results_names = ['loss', 'precision']
    test_results_names = ['loss', 'precision']

    os.makedirs(directory, exist_ok=True)

    logger = LoggerText(directory)
    logger.write_once(args)
    logger.write_once(train_results_names)
    logger.write_once(test_results_names)

    def train():
        avg_loss = AverageMeter()
        avg_accuracy = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        for step_id, data_train in enumerate(train_reader()):
            wrapped_results = exe.run(
                fine_tuning_program,
                feed=feeder_training.feed(data_train),
                fetch_list=[cost, accuracy, cur_lr])

            batch_time.update(time.time() - end)
            end = time.time()

            avg_loss.update(wrapped_results[0][0], len(data_train))
            avg_accuracy.update(wrapped_results[1][0], len(data_train))
            if step_id % 10 == 0:
                print(f"\tEpoch {e_id}, Step {step_id}, Batch_Time {batch_time.avg:.2f}, "
                      f"LR {wrapped_results[2][0]:.4f}, "
                      f"Loss {avg_loss.avg:.4f}, Acc {avg_accuracy.avg:.4f}", flush=True
                      )

        print(f"\tEpoch {e_id}, Step {step_id}, Batch_Time {batch_time.avg:.2f}, "
                      f"LR {wrapped_results[2][0]:.4f}, "
                      f"Loss {avg_loss.avg:.4f}, Acc {avg_accuracy.avg:.4f}"
                      )

        return avg_loss.avg, avg_accuracy.avg

    def test():
        avg_loss = AverageMeter()
        avg_accuracy = AverageMeter()

        for step_id, data_train in enumerate(test_reader()):
            avg_loss_value = exe.run(
                test_program,
                feed=feeder_test.feed(data_train),
                fetch_list=[cost, accuracy])
            avg_loss.update(avg_loss_value[0], len(data_train))
            avg_accuracy.update(avg_loss_value[1], len(data_train))

        print(f"Loss {avg_loss.avg}, Acc {avg_accuracy.avg}")

        return avg_loss.avg, avg_accuracy.avg

    # test_data = reader_creator_all_in_memory('./datasets/PetImages', is_test=True)
    for e_id in range(args.num_epoch):
        train_results = train()

        test_results = None
        if e_id % max(args.num_epoch // 10, 1) == 0 or e_id == args.num_epoch - 1:
            test_results = test()

        # logger
        logger.write_one_step_results(e_id, train_results, test_results)
        logger.flush()

        if e_id == args.num_epoch - 1:
            fluid.io.save_params(executor=exe, dirname=os.path.join(directory, f'snapshot'),
                                 main_program=fine_tuning_program)

        if args.fc_reinit > 0 and (e_id + 1) % (args.num_epoch // args.cyclic_num) == 0:
            if e_id > args.num_epoch - 5:
                continue
            for var in fine_tuning_program.all_parameters():
                if 'fc' in var.name:
                    print(var.name, var.shape)

                    var_training = fluid.global_scope().find_var(var.name).get_tensor()
                    print('************', fluid.global_scope().find_var(var.name).get_tensor()[0])
                    var_training.set(np.array(np.random.normal(0, np.sqrt(2/var.shape[0]), var.shape),  dtype=np.float32), place)
                    print(fluid.global_scope().find_var(var.name).get_tensor()[0])


if __name__ == '__main__':
    # args.dataset = 'PetImages'
    # args.pretrained_model = '/home/seven/codespace/pp_delta/pretrained_models/ResNet101_pretrained'
    print(args)
    main()

