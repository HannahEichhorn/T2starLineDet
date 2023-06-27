import logging
import os.path
import os
import torch.nn
import numpy as np
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import wandb
import plotly.graph_objects as go
from torchmetrics import Accuracy
from dl_utils import *
from core.DownstreamEvaluator import DownstreamEvaluator
from optim.losses.classification_losses import CrossEntropyAcrossLines


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)


    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.test_reconstruction(global_model)

    def test_reconstruction(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Reconstruction test #################")
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.model.load_state_dict(global_model)
        self.model.eval()

        # calculate different metrics on the datasets specified in the config file under downstream evaluation task
        metrics = {
            'BCE': [],
            'Accuracy': [],
            'FalseDetectedRate': [],
            'NotDetectedRate': []
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'BCE': [],
                'Accuracy': [],
                'FalseDetectedRate': [],
                'NotDetectedRate': []
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                with torch.no_grad():
                    # Input
                    kspace = data[0].to(self.device)
                    target_mask = data[1].to(self.device)
                    filename, slice_num = data[2:]

                    prediction = self.model(kspace)

                    for i in range(len(prediction)):
                        count = str(idx * len(prediction) + i)

                        # save the predicted masks for validation and test sets
                        if dataset_key in ['val', 'test', 'test_strong']:
                            if not os.path.exists(self.checkpoint_path+"/Predictions/"+dataset_key+"/"):
                                os.makedirs(self.checkpoint_path+"/Predictions/"+dataset_key+"/")
                            np.savetxt(self.checkpoint_path+"/Predictions/"+dataset_key+"/"+os.path.basename(filename[i])+"_Slice_"+str(slice_num[i].item())+".txt",
                                       prediction[i].detach().cpu().numpy())

                        # calculate binary cross entropy:
                        CELines = CrossEntropyAcrossLines()
                        tmp_target = target_mask[i]
                        tmp_pred = prediction[i]
                        loss_bce = CELines(tmp_target[None, :], tmp_pred[None, :])
                        test_metrics['BCE'].append(loss_bce.item())

                        # calculate accuracy:
                        accuracy = Accuracy(task='binary', threshold=0.5)
                        acc = accuracy(prediction[i].flatten().cpu(), target_mask[i].flatten().cpu().int())
                        test_metrics['Accuracy'].append(acc.item())

                        # threshold the prediction for the remaining metrics (Not detected and false detected rate):
                        thr_prediction = np.zeros_like(prediction[i].flatten().cpu())
                        thr_prediction[prediction[i].flatten().cpu() > 0.5] = 1

                        targ_0 = np.where(target_mask[i].flatten().cpu().int() == 0)[0]
                        pred_0 = np.where(thr_prediction == 0)[0]
                        false_det = [x for x in pred_0 if x not in targ_0]
                        not_det = [x for x in targ_0 if x not in pred_0]
                        test_metrics['FalseDetectedRate'].append((len(false_det)/len(thr_prediction)))
                        test_metrics['NotDetectedRate'].append((len(not_det) / len(thr_prediction)))


                        # log a few example images in wandb
                        if idx % 10 == 0 and i % 10 == 0:  # plot some images
                            prediction_ = prediction[i]

                            multiclass = False
                            if len(prediction_.shape) > 1:
                                multiclass = True
                                prediction_ = torch.argmax(prediction_, axis=0)

                            # Log one magnitude zero-filled and target image and reconstruction:
                            prediction_example = prediction_.detach().cpu().numpy().reshape(-1, 92)
                            target_example = target_mask[i].detach().cpu().numpy().reshape(-1, 92)

                            # reshape to full mask
                            prediction_example = np.rollaxis(np.tile(prediction_example, (112, 1, 1)), 0, 3)
                            target_example = np.rollaxis(np.tile(target_example, (112, 1, 1)), 0, 3)

                            # substitute two pixels to get consistent colormap:
                            prediction_example[0, 0, 0] = 0
                            prediction_example[0, 0, 1] = 1
                            target_example[0, 0, 0] = 0
                            target_example[0, 0, 1] = 1

                            if multiclass:
                                prediction_example = prediction_example / 4
                                target_example = target_example / 4

                            prediction_example = prediction_example[0]
                            target_example = target_example[0]

                            if len(np.unique(target_example)) < 3:
                                if not multiclass:
                                    thr_prediction = np.zeros_like(prediction_example)
                                    thr_prediction[prediction_example > 0.5] = 1

                            pred = wandb.Image(prediction_example[:, ::-1],
                                               caption='Predicted corruption mask')
                            targ = wandb.Image(target_example[:, ::-1], caption='Target corruption mask')
                            if not multiclass:
                                pred_th = wandb.Image(thr_prediction[:, ::-1],
                                                      caption='Predicted corruption mask (thresholded)')
                                wandb.log({"Reconstruction_Examples" + '_' + dataset_key + '/_' + str(count) + '_acc_' +
                                           str(acc.item()): [pred, pred_th, targ]})
                            else:
                                wandb.log({"Reconstruction_Examples" + '_' + dataset_key + '/_' + str(count) + '_acc_' +
                                           str(acc.item()): [pred, targ]})

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        # save the metrics
        for metric in metrics:
            for i, dataset in enumerate(self.test_data_dict.keys()):
                np.savetxt(self.checkpoint_path+"/"+metric+"_"+dataset+".txt", np.array(metrics[metric][i]).T,
                           header="Metric values for metric and dataset specified in filename")


        logging.info('Writing plots...')

        # plot the different metrics as box plots and log these to wandb
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Prediction_Metrics_" + self.name + '_' + str(metric): fig_bp})