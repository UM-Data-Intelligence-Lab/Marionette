import torch
import argparse
from evaluate_utils import get_task, get_run_data


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="marionette")
args = parser.parse_args()


def simulation(RUN_ID = "marionette", WANDB_DIR = "wandb", PROJECT_ROOT = "./"):
    # Get run data
    data_name, seed, run_path = get_run_data(RUN_ID, WANDB_DIR)
    # Get task and datamodule
    task, datamodule = get_task(run_path, data_root=PROJECT_ROOT)

    test_data = torch.load(PROJECT_ROOT+'data/'+data_name + f'/{data_name}_test.pkl')
    gps_dict = test_data['poi_gps']


    generated_seqs = []
    for (batch) in (datamodule.test_dataloader()): 
        time_samples = task.tpp_model.sample(batch.batch_size, tmax=batch.tmax.to(task.device), x_n=batch.to(task.device)).mask_check()
        assert len(time_samples) == batch.batch_size, "not enough samples"
        samples = task.discrete_diffusion.sample_fast(time_samples.to(task.device)).to_seq_list(gps_dict)
        assert len(samples) == batch.batch_size, "not enough samples"
        generated_seqs += samples


    data_new = {'sequences':generated_seqs,'t_max': batch.tmax.detach().cpu().numpy()}
    torch.save(data_new,f'./data/{data_name}/{data_name}_{RUN_ID}_generated.pkl')



simulation(RUN_ID=args.run_id)
