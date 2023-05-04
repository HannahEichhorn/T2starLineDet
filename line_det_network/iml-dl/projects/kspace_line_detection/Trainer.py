import os.path
from core.Trainer import Trainer
from time import time
import wandb
import logging
from torchinfo import summary
from optim.losses.image_losses import *


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        for s in self.train_ds:
            input_size = s[0].numpy().shape
            break
        dtypes = [torch.float64]
        print(f'Input size of summary is: {input_size}')
        summary(model, input_size, dtypes=dtypes)


    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            # load weights
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            # load optimizer
            self.optimizer.load_state_dict(opt_state)
        epoch_losses, epoch_losses_1, epoch_losses_2 = [], [], []
        self.early_stop = False
        self.model.train()

        for epoch in range(self.training_params['nr_epochs']):
            print('Epoch: ', epoch)
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_1, batch_loss_2, count_images = 0, 0, 0, 0

            for data in self.train_ds:
                # Input
                kspace = data[0].to(self.device)
                target_mask = data[1].to(self.device)

                count_images += kspace.shape[0]

                # Forward Pass
                self.optimizer.zero_grad()
                prediction = self.model(kspace)

                # Reconstruction Loss
                loss = self.criterion_rec(target_mask, prediction)

                # Backward Pass
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item() * kspace.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights


    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {task + '_loss_': 0}
        test_total = 0

        val_image_available = False
        with torch.no_grad():
            for data in test_data:
                kspace = data[0].to(self.device)
                target_mask = data[1].to(self.device)
                filename, slice_num = data[2:]

                test_total += kspace.shape[0]

                # Forward Pass
                prediction = self.test_model(kspace)
                prediction_track = prediction.clone().detach()
                target_mask_track = target_mask.clone().detach()

                loss_bce = self.criterion_rec(target_mask, prediction)

                metrics[task + '_loss_'] += loss_bce.item() * kspace.size(0)

                if task == 'Val':
                    # search for a specific validation image (if available)
                    search = [os.path.basename(f) + '_' + str(s.numpy()) for f, s in zip(filename, slice_num)]
                    if 'DATA_Epp_4_task-sub-p015_task-calc_acq-fullres_T2star_sim_b0_rigid_1.h5_15' in search:
                        # find out which:
                        ind = np.where(np.array(search) == 'DATA_Epp_4_task-sub-p015_task-calc_acq-fullres_T2star_sim_b0_rigid_1.h5_15')[0][0]
                        target_mask_ = target_mask_track[ind]
                        prediction_ = prediction_track[ind]
                        val_image_available = True

            if task == 'Val':
                if not val_image_available:
                    print('[Trainer - test] ERROR: No validation image can be tracked, since the required filename is '
                          'not in the test set')
                    print('Using the last available example instead')
                    target_mask_ = target_mask_track[0]
                    prediction_ = prediction_track[0]

                multiclass = False
                if len(prediction_.shape) > 1:
                    multiclass = True
                    prediction_ = torch.argmax(prediction_, axis=0)

                # Log one magnitude zero-filled and target image and reconstruction:
                prediction_example = prediction_.detach().cpu().numpy().reshape(-1, 92)
                target_example = target_mask_.detach().cpu().numpy().reshape(-1, 92)

                # reshape to full mask
                prediction_example = np.rollaxis(np.tile(prediction_example, (112,1,1)), 0, 3)
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

                pred = wandb.Image(prediction_example[:, ::-1], caption='Predicted corruption mask')
                targ = wandb.Image(target_example[:, ::-1], caption='Target corruption mask')
                if not multiclass:
                    pred_th = wandb.Image(thr_prediction[:, ::-1],
                                          caption='Predicted corruption mask (thresholded)')
                    wandb.log({task + '/Example_': [pred, pred_th, targ]})
                else:
                    wandb.log({task + '/Example_': [pred, targ]})

            for metric_key in metrics.keys():
                metric_name = task + '/' + str(metric_key)
                metric_score = metrics[metric_key] / test_total
                wandb.log({metric_name: metric_score, '_step_': epoch})
            wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
            epoch_val_loss = metrics[task + '_loss_'] / test_total
            if task == 'Val':
                print('Epoch: {} \tValidation Loss: {:.6f} , computed for {} samples'.format(
                    epoch, epoch_val_loss, test_total))
                if epoch_val_loss < self.min_val_loss:
                    self.min_val_loss = epoch_val_loss
                    torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                               self.client_path + '/best_model.pt')
                    self.best_weights = model_weights
                    self.best_opt_weights = opt_weights
                self.early_stop = self.early_stopping(epoch_val_loss)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch_val_loss)
