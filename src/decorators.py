import functools
import os
import torch


def train_device(current_gpu_index, model, dataloader, train_func, num_epochs, num_gpus):
    print(f"Training on GPU {current_gpu_index}")
    # Настройка группы процессов для текущего устройства
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device("cuda:{}".format(current_gpu_index))

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus,
        rank=current_gpu_index,
    )
    torch.cuda.set_device(device)

    # Определяем модель и перемещаем ее на текущее устройство
    model_device = model.to(current_gpu_index)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model_device,
        device_ids=[current_gpu_index],
        output_device=current_gpu_index
    )

    train_func(ddp_model, dataloader, device, num_epochs)

    # Очистка группы процессов
    torch.distributed.destroy_process_group()


def multi_gpu_training(train_func):
    @functools.wraps(train_func)
    def wrapper(model, dataloader, num_epochs=2):
        num_gpus = torch.cuda.device_count()
        print(f"Count GPU {num_gpus}")

        def train_device_wrapper(current_gpu_index):
            train_device(current_gpu_index, model, dataloader, train_func, num_epochs, num_gpus)

        torch.multiprocessing.start_processes(train_device_wrapper, args=(), nprocs=num_gpus, join=True,
                                              start_method="fork")

    return wrapper
