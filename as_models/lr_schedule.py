from paddle.fluid.layers import ops, tensor, learning_rate_scheduler
from paddle.fluid import data_feeder, framework
import math

def fc_cosine_decay(learning_rate, step_each_epoch, epochs, cyclic_num):
    data_feeder.check_type(learning_rate, 'learning_rate', (float, tensor.Variable),
               'fc_cosine_decay')

    with framework.default_main_program()._lr_schedule_guard():
        global_step = learning_rate_scheduler._decay_step_counter()

        cur_epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * 0.5 * (
            ops.cos(cur_epoch * math.pi / epochs) + 1)
        inverse = 1/decayed_lr
        u = epochs // cyclic_num
        fc_decayed_lr = learning_rate * 0.5 * (
            ops.cos((cur_epoch % u) * math.pi / u) + 1)
        ret = inverse * fc_decayed_lr
        #ret.name = 'fc_layer_lr'
        return ret

if __name__ == '__main__':
    res = fc_cosine_decay(0.1, 1000, 10, 3)
    print(res)
