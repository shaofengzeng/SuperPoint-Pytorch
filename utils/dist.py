import os
import torch
import torch.distributed as dist


def init_distributed_mode(config):
    param = {}
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        param.setdefault('rank', int(os.environ["RANK"]))
        param.setdefault('world_size', int(os.environ['WORLD_SIZE']))
        param.setdefault('gpu', int(os.environ['LOCAL_RANK']))
    elif 'SLURM_PROCID' in os.environ:
        param.setdefault('rank',int(os.environ['SLURM_PROCID']))
        param.setdefault('gpu',param['rank'] % torch.cuda.device_count())
    else:
        print('Not using distributed mode')
        param.setdefault('distributed',False)
        return

    param.setdefault('distributed',True)

    torch.cuda.set_device(param['gpu'])#set gpu id
    # 通信后端，nvidia GPU推荐使用NCCL
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=param['world_size'], rank=param['rank'])
    print('| distributed init (rank {}): env://'.format(param['rank']), flush=True)
    config['solver'].update(param)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value