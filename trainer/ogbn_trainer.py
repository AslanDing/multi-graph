import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import mi_loss,entropy_loss
from sklearn.metrics import f1_score

import nni

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_set, val_data_set=None,test_data_set=None,logger = None,
                 len_epoch=100,params=None):
        self.logger = logger
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device

        self.train_data_set = train_data_set
        self.val_data_set = val_data_set
        self.test_data_set = test_data_set

        self.batch_size = params['batch_size']
        self.len_epoch = len_epoch

        self.train_metrics = MetricTracker('loss', *['gib_log','gib_mi','pri','cross_mi'], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *['acc'], writer=self.writer)

        self.cross_entry = torch.nn.CrossEntropyLoss()

        self.PRI_beta = params['PRI_beta']
        self.PRI_alpha = params['PRI_alpha']
        self.PRI_weight = params['PRI_weight']

        self.GIB_beta = params['GIB_beta']
        self.GIB_cross_weight = params['GIB_cross_weight']

        self.length = len(self.train_data_set)

        self.do_validation = False

        self.log_step = 1

        self.max_microf1 = 0
        self.max_macrof1 = 0

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        iter_data = iter(self.train_data_set)
        for batch_idx in range(self.length//self.batch_size):  # self.length//self.batch_size
            loss_entry_all = 0
            pri_loss_all = 0
            mi_loss_multi = 0
            mi_loss_cross = 0

            domain_sub_graph_paper = []
            domain_sub_graph_author = []
            multi_graph_src_emb = []
            multi_graph_sub_emb = []

            for i in range(self.batch_size):
                data = next(iter_data)
                data = data.to(self.device)
                x_dict = data.x_dict
                edge_dict = data.edge_index_dict
                batch_dict = data.batch_dict
                y_dict = data.y_dict
                mask_dict = data.mask_dict

                y_hat, w_dict, \
                multi_embedding_pool, multi_src_embedding_pool, \
                node_embedding_sub = self.model(x_dict, edge_dict,batch_dict,mask_dict['paper'])

                # cal loss
                # entry
                loss_entry = self.cross_entry(y_hat, y_dict['paper'])
                loss_entry_all += loss_entry

                # H(G_s)
                num_start_paper = 0
                num_start_author = 0
                num_start_psp = 0
                num_start_afa = 0
                loss_pri_all = 0
                loss_pri_paper = 0
                loss_pri_author = 0
                for batch_index in range(1):
                    # how many nums
                    num_paper = mask_dict['paper'][batch_index].item()
                    num_author = mask_dict['author'][batch_index].item()
                    num_psp = mask_dict[('paper', 'subject', 'paper')][batch_index].item()
                    num_afa = mask_dict[('author', 'Institute', 'author')][batch_index].item()

                    paper_node_num = num_paper
                    paper_paper_edge_num = num_psp
                    E_paper_list = [[], []]
                    E_paper_value = []
                    E_d_paper_list = [[], []]
                    E_d_paper_value = []
                    for index in range(num_start_psp, num_psp + num_start_psp):
                        first_node = edge_dict[('paper', 'subject', 'paper')][0, index]
                        second_node = edge_dict[('paper', 'subject', 'paper')][1, index]
                        E_paper_list[0].append(first_node - num_start_paper)  # start node
                        E_paper_list[1].append(index - num_start_psp)         # edge
                        E_paper_value.append(-1.0)
                        E_paper_list[0].append(second_node - num_start_paper) # end node
                        E_paper_list[1].append(index - num_start_psp)         # edge
                        E_paper_value.append(1.0)

                        # value = w_dict[('paper', 'subject', 'paper')][index]
                        # E_d_paper_list[0].append(index - num_start_psp)
                        # E_d_paper_list[1].append(index - num_start_psp)
                        # E_d_paper_value.append(value)

                    if paper_node_num <= 1 or paper_paper_edge_num <= 1:
                        loss_pri_paper = 0
                    else:
                        E_paper = torch.sparse_coo_tensor(torch.tensor(E_paper_list),
                                                          E_paper_value,
                                                          (paper_node_num, paper_paper_edge_num)).float().to(self.device)
                        diag_paper = torch.diag(w_dict[('paper', 'subject', 'paper')])
                        # diag_paper = torch.sparse_coo_tensor(torch.tensor(E_d_paper_list),
                        #                                      E_d_paper_value,
                        #                                      (paper_paper_edge_num, paper_paper_edge_num)).float()
                        rho_paper = E_paper @ E_paper.t()
                        tmp = torch.sparse.mm(E_paper, diag_paper)
                        sigma_paper = torch.sparse.mm(E_paper,tmp.transpose(1,0))
                        loss_pri_paper = entropy_loss(sigma_paper.to_dense(), rho_paper.to_dense().detach(), self.PRI_beta, self.PRI_alpha)
                        loss_pri_paper = torch.relu(loss_pri_paper)
                    author_node_num = num_author
                    author_author_edge_num = num_afa
                    E_author_list = [[], []]
                    E_author_value = []
                    # E_d_author_list = [[], []]
                    # E_d_author_value = []
                    for index in range(num_start_afa, num_afa + num_start_afa):
                        first_node = edge_dict[('author', 'Institute', 'author')][0, index]
                        second_node = edge_dict[('author', 'Institute', 'author')][1, index]
                        E_author_list[0].append(first_node - num_start_author)
                        E_author_list[1].append(index - num_start_afa)
                        E_author_value.append(-1.0)
                        E_author_list[0].append(second_node - num_start_author)
                        E_author_list[1].append(index - num_start_afa)
                        E_author_value.append(1.0)

                        # value = w_dict[('author', 'Institute', 'author')][index]
                        # E_d_author_list[0].append(index - num_start_afa)
                        # E_d_author_list[1].append(index - num_start_afa)
                        # E_d_author_value.append(value)

                    if author_node_num <= 1 or author_author_edge_num <= 1:
                        loss_pri_author = 0
                    else:
                        E_author = torch.sparse_coo_tensor(torch.tensor(E_author_list),
                                                           E_author_value,
                                                           (author_node_num, author_author_edge_num)).float().to(self.device)
                        diag_author = torch.diag(w_dict[('author', 'Institute', 'author')])
                        # diag_author = torch.sparse_coo_tensor(torch.tensor(E_d_author_list),
                        #                                       E_d_author_value,
                        #                                       (author_author_edge_num, author_author_edge_num)).float()
                        rho_author = E_author @ E_author.t()
                        tmp = torch.sparse.mm(E_author, diag_author)
                        sigma_author = torch.sparse.mm(E_author, tmp.transpose(1, 0))
                        #sigma_author = E_author @ diag_author @ E_author.t()
                        loss_pri_author = entropy_loss(sigma_author.to_dense(), rho_author.to_dense().detach(), self.PRI_beta,
                                                       self.PRI_alpha)
                        loss_pri_author = torch.relu(loss_pri_author)
                    loss_pri = loss_pri_paper + loss_pri_author
                    loss_pri_all += loss_pri
                pri_loss_all += loss_pri_all

                domain_sub_graph_paper.append(node_embedding_sub['paper'])
                domain_sub_graph_author.append(node_embedding_sub['author'])
                multi_graph_sub_emb.append(multi_embedding_pool)
                multi_graph_src_emb.append(multi_src_embedding_pool.detach())

            # mi loss
            for i in range(self.batch_size):
                query_vector = multi_graph_sub_emb[i]
                positve_vector = multi_graph_src_emb[i]
                mi_loss_tmp = mi_loss(query_vector,positve_vector,multi_graph_src_emb,i)
                mi_loss_multi += torch.relu(mi_loss_tmp)

                query_vector = domain_sub_graph_paper[i]
                positve_vector = domain_sub_graph_author[i]
                mi_loss_tmp = mi_loss(query_vector, positve_vector, multi_graph_src_emb, i)
                mi_loss_cross += torch.relu(mi_loss_tmp)

            loss_entry_all /= self.batch_size
            pri_loss_all /= self.batch_size
            mi_loss_multi /= self.batch_size
            mi_loss_cross /= self.batch_size

            total_loss = loss_entry_all + self.GIB_beta*mi_loss_multi + \
                         self.PRI_weight*pri_loss_all + \
                         self.GIB_cross_weight*mi_loss_cross

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_set) + batch_idx, 'train')
            try:
                total_loss = total_loss.item()
            except Exception:
                total_loss = total_loss

            try:
                loss_entry_all = loss_entry_all.item()
            except Exception:
                loss_entry_all = loss_entry_all

            try:
                mi_loss_multi = mi_loss_multi.item()
            except Exception:
                mi_loss_multi = mi_loss_multi

            try:
                pri_loss_all = pri_loss_all.item()
            except Exception:
                pri_loss_all = pri_loss_all

            try:
                mi_loss_cross = mi_loss_cross.item()
            except Exception:
                mi_loss_cross = mi_loss_cross

            self.train_metrics.update('loss', total_loss)
            self.train_metrics.update('gib_log', loss_entry_all)
            self.train_metrics.update('gib_mi', mi_loss_multi)
            self.train_metrics.update('cross_mi', pri_loss_all)
            self.train_metrics.update('cross_mi', mi_loss_cross)

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f},gib_log: {:.6f},'
                                 'gib_mi: {:.6f},pri: {:.6f},'
                                 'cross_mi: {:.6f}'.format(
                    epoch, self._progress(batch_idx),
                    total_loss, loss_entry_all,
                    mi_loss_multi, pri_loss_all,
                    mi_loss_cross))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            y_hats = []
            y_labels = []
            acc = []
            for batch_idx, data in enumerate(self.val_data_set):

                data = data.to(self.device)
                x_dict = data.x_dict
                edge_dict = data.edge_index_dict
                batch_dict = data.batch_dict
                y_dict = data.y_dict
                mask_dict = data.mask_dict

                y_hat, w_dict  = self.model(x_dict, edge_dict,batch_dict,mask_dict['paper'])

                acc.append((torch.argmax(y_hat, 1) == torch.argmax(data.y_dict['paper'], 1)).cpu().detach().numpy())
                y_hats.append(torch.argmax(y_hat, 1).cpu().detach().numpy())
                y_labels.append(torch.argmax(data.y_dict['paper'], 1).cpu().detach().numpy())
            acc = np.array(acc).mean()
            self.writer.set_step((epoch - 1) * len(self.val_data_set) + batch_idx, 'valid')
            self.valid_metrics.update('acc', acc)
            micre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="micro")
            macre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="macro")

            if self.max_microf1<micre_f1:
                self.max_microf1 = micre_f1
            if self.max_macrof1<macre_f1:
                self.max_macrof1 = macre_f1
            self.logger.info(' acc ' + str(acc) + " micro_f1 " + str(micre_f1) + " macro_f1 " + str(macre_f1))
            self.logger.info(' best ' + " micro_f1 " + str(self.max_microf1) + " macro_f1 "+ str(self.max_macrof1))

            nni.report_intermediate_result(micre_f1)
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        #self._test_epoch()
        return self.valid_metrics.result()

    def _test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            y_hats = []
            y_labels = []
            acc = []
            for batch_idx, data in enumerate(self.test_data_set):
                data = data.to(self.device)
                x_dict = data.x_dict
                edge_dict = data.edge_index_dict
                batch_dict = data.batch_dict
                y_dict = data.y_dict
                mask_dict = data.mask_dict

                y_hat, w_dict = self.model(x_dict, edge_dict, batch_dict,mask_dict['paper'])

                acc.append((torch.argmax(y_hat, 1) == torch.argmax(data.y_dict['paper'], 1)).cpu().detach().numpy())
                y_hats.append(torch.argmax(y_hat, 1).cpu().detach().numpy())
                y_labels.append(torch.argmax(data.y_dict['paper'], 1).cpu().detach().numpy())

            micre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="micro")
            macre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="macro")
            acc = np.array(acc).mean()

            if self.max_microf1<micre_f1:
                self.max_microf1 = micre_f1
            if self.max_macrof1<macre_f1:
                self.max_macrof1 = macre_f1
            self.logger.info(' acc '+ str(acc) + " micro_f1 " + str(micre_f1) + " macro_f1 "+ str(macre_f1))
            self.logger.info(' best ' + " micro_f1 " + str(self.max_microf1) + " macro_f1 "+ str(self.max_macrof1))

            nni.report_final_result(self.max_microf1)
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'

        current = batch_idx * self.batch_size
        total = self.length

        return base.format(current, total, 100.0 * current / total)
