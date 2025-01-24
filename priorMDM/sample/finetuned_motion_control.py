# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from model.cfg_sampler import wrap_model
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml_utils import get_inpainting_mask
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_multi_3d_motion_extended
import shutil

def main():
    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')

    """##############################################################"""
    synthetic_data = np.load('/home/ML_courses/03683533_2024/anton_kfir_daniel/priorMDM-Trace/integration/synthetic_data.npy')
    init_positions = np.load('/home/ML_courses/03683533_2024/anton_kfir_daniel/priorMDM-Trace/integration/init_positions.npy')
    args.batch_size = synthetic_data.shape[0]
    args.num_samples = synthetic_data.shape[0]
    """##############################################################"""

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              load_mode='train',
                              size=args.num_samples)  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())

    """##############################################################"""
    # normalization
    mean = data.dataset.t2m_dataset.mean[:, None, None]  
    std = data.dataset.t2m_dataset.std[:, None, None]   

    synthetic_data = (synthetic_data -  mean) / std

    synthetic_data = torch.from_numpy(synthetic_data).float()

    input_motions = synthetic_data.to(dist_util.dev())

    model_kwargs['y']['lengths'] = torch.full((args.batch_size,), max_frames)
    model_kwargs['y']['mask'][:, :, :, :4] = True
    model_kwargs['y']['mask'][:, :, :, 5:] = False
    model_kwargs['y']['text'] = ["walk with one arm raized" for _ in range(args.batch_size)]

    print("synthetic_data was successfully injected into the model.")
    """##############################################################"""


    # if args.text_condition != '':
    #     texts = [args.text_condition] * args.num_samples
    #     model_kwargs['y']['text'] = texts

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints) # recover
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    """##############################################################"""
    all_motions_list = []
    all_titles_list = []

    # Instead of looping and plotting inside that loop,
    # we only gather everything:
    for sample_i in range(args.num_samples):

        # If you want to show the input:
        if args.show_input:
            caption = f'Input Motion (sample {sample_i})'
            length = model_kwargs['y']['lengths'][sample_i]
            motion_input = input_motions[sample_i].transpose(2, 0, 1)[:length]
            all_motions_list.append(motion_input)
            all_titles_list.append(caption)

        # Now gather the repeated motions:
        for rep_i in range(args.num_repetitions):
            raw_caption = all_text[rep_i * args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = f'Edit [{args.inpainting_mask}] unconditioned (sample {sample_i}, rep {rep_i})'
            else:
                caption = f'Edit [{args.inpainting_mask}]: {raw_caption} (sample {sample_i}, rep {rep_i})'
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion_rep = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]

            all_motions_list.append(motion_rep)
            all_titles_list.append(caption)

    # Now we have one giant list containing all skeletons
    # for all samples and all reps.

    # 1) 'default' view
    out_filename_default = 'all_samples_default.mp4'
    output_path_default = os.path.join(out_path, out_filename_default)
    plot_multi_3d_motion_extended(
        save_path=output_path_default,
        kinematic_tree=skeleton,
        motions_list=all_motions_list,
        init_coordinations=init_positions,
        titles=all_titles_list,
        dataset=args.dataset,
        fps=fps,
        color_mode='multi',
        camera_view='default'
    )

    # 2) 'top' (true top-down) view
    out_filename_top = 'all_samples_top.mp4'
    output_path_top = os.path.join(out_path, out_filename_top)
    plot_multi_3d_motion_extended(
        save_path=output_path_top,
        kinematic_tree=skeleton,
        motions_list=all_motions_list,
        init_coordinations=init_positions,
        titles=all_titles_list,
        dataset=args.dataset,
        fps=fps,
        color_mode='multi',
        camera_view='top'
    )

    # 3) 'side' view
    out_filename_side = 'all_samples_side.mp4'
    output_path_side = os.path.join(out_path, out_filename_side)
    plot_multi_3d_motion_extended(
        save_path=output_path_side,
        kinematic_tree=skeleton,
        motions_list=all_motions_list,
        init_coordinations=init_positions,
        titles=all_titles_list,
        dataset=args.dataset,
        fps=fps,
        color_mode='multi',
        camera_view='side'
    )

    print('[Done] Saved three separate MP4 files:')
    print('   1) ' + output_path_default)
    print('   2) ' + output_path_top)
    print('   3) ' + output_path_side)
    """##############################################################"""



if __name__ == "__main__":
    main()