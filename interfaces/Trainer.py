import torch
import os
from datetime import datetime
import copy
import torch.nn.functional as F
import cv2
from interfaces import base
from utils.util import str_filt
import numpy as np
from ptflops import get_model_complexity_info
import time
import logging
from model.parseq.parseq_LR import parse_info_generator

class TextSR(base.TextBase):
    def train(self):
        cfg = self.config.TRAIN

        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()

        HR_teacher = self.HRteacher_init()
        model_dict = self.generator_init()

        LR_student, image_crit = model_dict['model'], model_dict['crit']
        LR_student.train()

        optimizer_G = self.optimizer_init(LR_student,image_crit)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G,25, 0.1)

        best_acc = 0
        converge_list = []
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)

        for epoch in range(cfg.epochs):
            for j, data in (enumerate(train_loader)):
                iters = len(train_loader) * epoch + j + 1
                if not self.args.go_test:
                    for p in LR_student.parameters():
                        p.requires_grad = True

                    images_hr, images_lr, gt_dict, label_strs = data
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)
                    gt_dict['label'] = gt_dict['label'].to(self.device)
                    gt_dict['length'] = gt_dict['length'].to(self.device).unsqueeze(1)

                    common_info = None
                    if self.args.rec_backbone in ['PARSeq']:
                        common_info = parse_info_generator(self.charsetMapper,label_strs,self.device)

                    output_stu = LR_student(images_lr,common_info=common_info)
                    with torch.no_grad():
                        output_tea = HR_teacher(images_hr,common_info=common_info)

                    loss, visual_loss, sematic_loss, ce_loss, kl_loss = image_crit(output_stu, output_tea, gt_dict)

                    optimizer_G.zero_grad()
                    loss = loss.mean()  # multi-gpu
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(LR_student.parameters(), 0.25)
                    optimizer_G.step()

                    if iters % cfg.displayInterval == 0:
                        logging.info('[{}]\t'
                                     'Epoch: [{}][{}/{}]\t'
                                     'total_loss {:.3f}\t'
                                     'visual_loss {:.5f}\t'
                                     'sematic_loss {:.3f}\t'
                                     'ce_loss {:.3f}\t'
                                     'kl_loss {:.3f}\t'
                                     .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                             epoch, j + 1, len(train_loader),
                                             float(loss.data),
                                             visual_loss.mean().item(),
                                             sematic_loss.mean().item(),
                                             ce_loss.mean().item(),
                                             kl_loss.mean().item()
                                             ))

                if iters % cfg.VAL.valInterval == 0 or self.args.go_test:
                    logging.info('======================================================')
                    current_acc_dict = {}

                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)

                        LR_student.eval()
                        for p in LR_student.parameters():
                            p.requires_grad = False

                        metrics_dict = self.eval(
                            LR_student,
                            HR_teacher,
                            val_loader)

                        for p in LR_student.parameters():
                            p.requires_grad = True
                        LR_student.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if self.args.go_test:
                        exit()
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        logging.info('saving best model')
                        logging.info('avg_acc {:.4f}%'.format(100 * (current_acc_dict['easy'] * 1619
                                                                     + current_acc_dict['medium'] * 1411
                                                                     + current_acc_dict['hard'] * 1343) / (
                                                                          1343 + 1411 + 1619)))
                        logging.info('=============')
                        self.save_checkpoint(LR_student, best_model_acc, True)

                if iters % cfg.saveInterval == 0:
                    self.save_checkpoint(LR_student, best_model_acc, False)

            lr_scheduler.step()

    def eval(self, stu_model, tea_model, val_loader):
        n_correct = 0  # stu model
        n_correct_lr = 0  # tea model with LR input
        n_correct_hr = 0
        sum_images = 0  # total images in the val loader
        metric_dict = {'accuracy': 0.0}
        filter_mode = 'lower'
        infer_time = 0

        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, gt_dict, label_strs = data
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            gt_dict['label'] = gt_dict['label'].to(self.device)
            gt_dict['length'] = gt_dict['length'].to(self.device)

            before = time.time()
            stu_output = stu_model(images_lr)
            after = time.time()
            infer_time += (after - before)

            tea_output_LR = tea_model(images_lr, input_lr=True)
            tea_output_HR = tea_model(images_hr, input_lr=False)


            string_stu = self.charsetMapper.logits_to_string(stu_output)
            string_tea_LR = self.charsetMapper.logits_to_string(tea_output_LR)
            string_tea_HR = self.charsetMapper.logits_to_string(tea_output_HR)

            for batch_i in range(images_lr.shape[0]):

                label = label_strs[batch_i]

                if str_filt(string_stu[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct += 1

                if str_filt(string_tea_LR[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_lr += 1

                if str_filt(string_tea_HR[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_hr += 1

            sum_images += images_lr.shape[0]
            torch.cuda.empty_cache()


        accuracy = round(n_correct / sum_images, 4)
        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)

        logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        logging.info('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        logging.info('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        metric_dict['accuracy'] = accuracy

        inference_time = sum_images / infer_time
        logging.info("AVG inference:Per second process {} images".format(inference_time))
        logging.info("sum_images:{}".format(sum_images))

        return metric_dict
