"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from uuid import uuid4

import fastargs
import numpy as np
import torch

import datasets
import dent.models
import dent.losses
import dent.losses.utils
import dent.utils
import parameters
import utils
import utils.visualize

def main(config):
    # Get logger.
    logger = utils.set_logger(__name__)
    # Set correct seed.
    utils.set_seed()
    # Get and set correct device.
    device = utils.get_device()
    # Base flags - separate training, evaluation & visualisation.
    run_training = 'train' in config['run.do']
    run_evaluation = 'eval' in config['run.do']
    run_visualization = 'visualize' in config['run.do']

    # Set all required folders and files if not explicitly set.
    config, info_dict = utils.set_save_paths_or_restore(config, logger, device)
    config.summary()

    if run_training:
        train_log_summary = {}
        constraints_filepath = config['constraints.file']
        if constraints_filepath == 'none':
            constraints_filepath = None
        correlations_filepath = config['constraints.correlations_file']
        if correlations_filepath == 'none':
            correlations_filepath = None
        train_loader, log_summary = datasets.get_dataloaders(
            shuffle=True, 
            device=device, 
            logger=logger, 
            return_pairs='pairs' in config['train.supervision'], 
            root=config['data.root'],
            k_range=config['data.k_range'], 
            constraints_filepath=constraints_filepath,
            correlations_filepath=correlations_filepath)
        config = utils.insert_config('data.img_size',
                                     train_loader.dataset.img_size)
        config = utils.insert_config('data.background_color',
                                     train_loader.dataset.background_color)
        log_summary['data.img_size'] = train_loader.dataset.img_size
        log_summary['data.background_color'] = train_loader.dataset.background_color
        train_log_summary.update({f'train_loader.{key}': item for key, item in log_summary.items()})

        model = dent.model_select(device, img_size=config['data.img_size'])
        train_log_summary.update({
            'model.parameter_count_total': sum(p.numel() for p in model.parameters()),
            'model.parameter_count_train': sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

        if hasattr(model, 'to_optim'):
            optimizer = utils.get_optimizer(model.to_optim)
        else:
            optimizer = utils.get_optimizer(model.parameters())
        scheduler = utils.get_scheduler(optimizer)
        model = model.to(device)
        loss_f = dent.loss_select(device, n_data=len(train_loader.dataset))
        if config['train.supervision'] == 'none':
            trainer = dent.UnsupervisedTrainer(
                model=model, optimizer=optimizer, scheduler=scheduler, loss_f=loss_f,
                device=device, logger=logger, read_dir=info_dict['read_dir'],
                write_dir=info_dict['write_dir'])
        elif 'pairs' in config['train.supervision']:
            infer_k = True
            if config['train.supervision'] == 'pairs_and_shared':
                infer_k = False
            trainer = dent.WeaklySupervisedPairTrainer(
                model=model, optimizer=optimizer, scheduler=scheduler, loss_f=loss_f,
                device=device, logger=logger, read_dir=info_dict['read_dir'],
                write_dir=info_dict['write_dir'], infer_k=infer_k)
        else:
            raise ValueError(f'No trainer [{config["train.supervision"]}] available!')

        #Initialize main trainer class. Note that W&B is initialized here.
        if info_dict['start_from_chkpt']:
            uid = None if not config['run.restore_to_new'] else str(uuid4())            
            trainer.initialize_from_checkpoint(
                info_dict['chkpt_data'], uid=uid)
        else:
            trainer.initialize()

        #Update & Storey Run Summary information (in W&B if available).
        trainer.data_logger.log_summary(train_log_summary, log_key='train')

        #If pair-based weak supervision is used: visualize and log example sample pairs.
        if config['viz.plot_sample_pairs'] and 'pairs' in config['train.supervision']:
            utils.visualize.plot_pair_samples(
                train_loader, write_dir=info_dict['write_dir'], log_to_wandb=config['log.wandb_mode']!='off')

        trainer(train_loader)
    else:
        logger.info('Skipped Training.')


    if run_evaluation or run_visualization:
        if run_training:
            data_logger = trainer.data_logger
        else:
            import datalogger
            if info_dict['chkpt_data'] is None:
                raise Exception(f"Can't evaluate - no checkpoints available at {info_dict['read_dir']}!")
            uid = info_dict['chkpt_data']['uid'] if not config['run.restore_to_new'] else str(uuid4())            
            data_logger = datalogger.DataLogger(info_dict['write_dir'], uid=uid)

        #Evaluate/Visualize every checkpoint on the same dataloader.
        test_loader, _ = datasets.get_dataloaders(shuffle=False,
                                                  batch_size=config['eval.batch_size'],
                                                  device=device,
                                                  logger=logger,
                                                  root=config['data.root'])
        if config['eval.mode'] == 'all':
            #Evaluate on all logged checkpoints "chkpt-N.pth.tar" (train.checkpoint_every) in sequence.
            chkpts = [x for x in os.listdir(info_dict['read_dir']) if '.pth.tar' in x and '-' in x]
            chkpts = sorted(chkpts, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        elif config['eval.mode'] == 'last':
            #Evaluate only on the last checkpoint ("chkpt.pth.tar").
            chkpts = ['chkpt.pth.tar']
        elif config['eval.mode'] == 'exact':
            #Provide a path to a specific checkpoint to utilize.
            chkpts = ['use_existing']
        else:
            eval_modes = ['all', 'last', 'exact']
            raise ValueError(
                'eval.mode = {} does not match {}'.format(
                    config['eval.mode'], eval_modes))

        #Iterate over all selected checkpoints.
        PLOT_NAMES = utils.visualize.PLOT_NAMES
        total_evaluation_results = {}
        for chkpt in chkpts:
            logger.info(f'Working on chkpt: {chkpt}...')
            if chkpt == 'use_existing':
                eval_model = dent.model_select(
                    device=device, name=config['model.name'], img_size=config['data.img_size'])
                eval_loss_f = dent.loss_select(device,
                                               name=config['train.loss'],
                                               n_data=len(test_loader.dataset))
            else:
                eval_model, info_dict['chkpt_data'], config = dent.utils.io.load_checkpoint(
                    info_dict['read_dir'], device, chkpt_name=chkpt, overwrite=config['run.restore_with_config'])
                eval_loss_f = dent.loss_select(device,
                                               name=info_dict['chkpt_data']['metadata']['train.loss'],
                                               n_data=len(test_loader.dataset))
            eval_model.eval()

            #Compute metric/loss on checkpoints.
            if run_evaluation:
                logger.info('Evaluating...')
                evaluator = dent.Evaluator(device=device, logger=logger)
                eval_out = evaluator(eval_model, test_loader, eval_loss_f)
                total_evaluation_results[chkpt] = eval_out
                logger.info(f'Eval. metrics: {eval_out}')
                eval_out['chkpt_epoch'] = info_dict['chkpt_data']['epoch']
                data_logger.log(eval_out, log_key='eval')

            #Visualize each checkpoint.
            if run_visualization:
                logger.info('Visualizing...')
                # TODO: Is overwriting happening here as well?
                viz = utils.visualize.Visualizer(
                    model=eval_model, write_dir=info_dict['write_dir'], loss_of_interest=None)
                size = (config['viz.nrows'], config['viz.ncols'])
                num_samples = np.prod(size)
                samples = utils.visualize.get_samples(
                    test_loader.dataset, num_samples, logger=logger, idcs=config['viz.idcs'])

                # Compute KL-Divergence to Normal for each latent (averaged over drawn samples).
                loss_per_latent = None
                if config['viz.reorder_latents_by_kl']:
                    with torch.no_grad():
                        ev_out = eval_model.encoder(samples.to(device))
                    loss_per_latent = dent.losses.utils._kl_normal_loss(
                        *ev_out['stats_qzx'].unbind(-1), return_components=True)
                    loss_per_latent = loss_per_latent.detach().cpu().numpy()

                if "all" in config['viz.plots']:
                    config.collect({
                        'viz.plots': [
                            'generate-samples', 'data-samples', 'reconstruct',
                            "traversals", 'reconstruct-traverse', "gif-traversals"
                        ]
                    })

                utils.visualize.PLOT_NAMES = {
                    key: item.replace('.', f"_{chkpt.replace('.pth.tar', '')}.") for key, item in PLOT_NAMES.items()}

                # return output and upload to W&B.
                image_dict = {}

                for plot_type in config['viz.plots']:
                    logger.info(f'Generating plot [{plot_type}].')
                    if plot_type == 'generate-samples':
                        return_dict = viz.generate_samples(size=size)
                    elif plot_type == 'data-samples':
                        return_dict = viz.data_samples(samples, size=size)
                    elif plot_type == "reconstruct":
                        return_dict = viz.reconstruct(samples, size=size)
                    elif plot_type == 'traversals':
                        return_dict = viz.traversals(
                            data=samples[0:1, ...] if config['viz.is_posterior'] else None,
                            n_per_latent=config['viz.ncols'],
                            n_latents=config['viz.nrows'],
                            loss_per_latent=loss_per_latent)
                    elif plot_type == "reconstruct-traverse":
                        return_dict = viz.reconstruct_traverse(samples,
                                                is_posterior=config['viz.is_posterior'],
                                                n_latents=config['viz.nrows'],
                                                n_per_latent=config['viz.ncols'],
                                                is_show_text=config['viz.show_loss'],
                                                loss_per_latent=loss_per_latent)
                    elif plot_type == "gif-traversals":
                        return_dict = viz.gif_traversals(
                            samples[:config['viz.ncols'], ...], 
                            n_latents=config['viz.nrows'],
                            loss_per_latent=loss_per_latent)
                    else:
                        raise ValueError("Unkown plot_type={}".format(plot_type))
                    image_dict[plot_type] = return_dict['path']
                    image_dict['chkpt_epoch'] = info_dict['chkpt_data']['epoch']                    
                    data_logger.log(image_dict, is_img='all-chkpt_epoch', log_key='viz')
        
        #Print and save evaluation summary:
        if run_evaluation:
            summary = '\nCheckpoints:\n'+ ' | '.join(f'({i+1}) {key}' for i, key in enumerate(total_evaluation_results.keys()))
            metrics = sorted(list(total_evaluation_results[chkpts[0]]['metrics'].keys()))
            for metric in metrics:
                summary += '\n\n'
                summary += f'{metric}:\n'
                for i, chkpt in enumerate(chkpts):
                    if i>0:
                        summary += ' | '
                    summary += f'({i+1}) ' + '{0:2.5f}'.format(total_evaluation_results[chkpt]['metrics'][metric])
            summary += '\n'        
            with open(info_dict['write_dir']/'evaluation_results.txt', 'w') as file:
                file.write(summary)
            logger.info(summary)
    else:
        if not run_evaluation:
            logger.info('Skipped Evaluation.')
        if not run_visualization:
            logger.info('Skipped Visualization.')


if __name__ == '__main__':
    # Load input arguments (via argparse).
    utils.make_config()
    # Get current parameter configuration.
    config = fastargs.get_current_config()    
    main(config)
