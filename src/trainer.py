import os
import math
import mmap
from decimal import Decimal
import time

import utility
import imageio

import torch
import torch.nn.utils as utils
from tqdm import tqdm


def get_list_num_with_write(datalist, newlist, result_file):
    with open(result_file, 'a') as f:
        for i in datalist:
            if isinstance(i, list):
                get_list_num_with_write(i, newlist, result_file)
            else:
                newlist.append(i)
                f.write(str(i))
                f.write('\n')
    f.close()


def get_list_num(datalist, newlist):
    for i in datalist:
        if isinstance(i, list):
            get_list_num(i, newlist)
        else:
            newlist.append(i)


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        d_gap = 128
        width = 4096 / d_gap
        height = 2048 / d_gap
        list_visited = []

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        mmap_file = mmap.mmap(-1, 67108864, access=mmap.ACCESS_WRITE, tagname='sharemem')
        sr_mmap_file = mmap.mmap(-1, 40960, access=mmap.ACCESS_WRITE, tagname='sr')
        loop_count = '-1'

        result_file_name = 'results.txt'

        timer_test = utility.timer()
        data_dic_lr = {}
        data_dic_hr = {}
        # if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                print(self.loader_test)
                d.dataset.set_scale(idx_scale)
                for lr_ori, hr_ori, filename_ori in d:
                    data_dic_lr[filename_ori[0]] = lr_ori
                    data_dic_hr[filename_ori[0]] = hr_ori
                print("Preprocess Completed!")
                # for lr, hr, filename in tqdm(d, ncols=200):

                partition_mmap_file = mmap.mmap(-1, 40960, access=mmap.ACCESS_WRITE, tagname='partition')
                while_loop = 1
                while while_loop == 1:
                    partition_mmap_file.seek(0)
                    mmap_partition_num_0 = int(partition_mmap_file.read_byte())
                    # if mmap_partition_num_0 != 0:
                    #     print(mmap_partition_num_0)
                    mmap_partition_num_1 = int(partition_mmap_file.read_byte())
                    # if mmap_partition_num_1 != 0:
                    #     print(mmap_partition_num_1)
                    mmap_partition_num_2 = int(partition_mmap_file.read_byte())
                    # if mmap_partition_num_2 != 0:
                    #     print(mmap_partition_num_2)
                    mmap_partition_num_3 = int(partition_mmap_file.read_byte())
                    # if mmap_partition_num_3 != 0:
                    #     print(mmap_partition_num_3)
                    mmap_partition_num = str(
                        1000 * mmap_partition_num_0 + 100 * mmap_partition_num_1 + 10 * mmap_partition_num_2 + 1 * mmap_partition_num_3 - 1)
                    if mmap_partition_num != loop_count and mmap_partition_num != '-1':
                        list_cand = []
                        num_center = int(mmap_partition_num)
                        if num_center not in list_visited:
                            list_cand.append(num_center)
                            list_visited.append(num_center)

                        num_top = num_center - width
                        if num_top >= 0 and num_top not in list_visited:
                            list_cand.append(num_top)
                            list_visited.append(num_top)

                        num_bottom = num_center + width
                        if num_bottom < width * height and num_bottom not in list_visited:
                            list_cand.append(num_bottom)
                            list_visited.append(num_bottom)

                        num_left = num_center - 1
                        if num_left <= int(num_center / width) * width and num_left not in list_visited:
                            list_cand.append(num_left)
                            list_visited.append(num_left)
                        else:
                            new_partition_num = (int(num_center / width) + 1) * width - 1
                            if new_partition_num not in list_visited:
                                list_cand.append(new_partition_num)
                                list_visited.append(new_partition_num)


                        num_right = num_center + 1
                        if num_right < (int(num_center / width) + 1) * width and num_right not in list_visited:
                            list_cand.append(num_right)
                            list_visited.append(num_right)
                        else:
                            new_partition_num = int(num_center / width) * width
                            if new_partition_num not in list_visited:
                                list_cand.append(new_partition_num)
                                list_visited.append(new_partition_num)

                        num_top_left = num_center - width - 1
                        if num_top_left >= 0:
                            if num_top_left <= int(num_top / width) * width and num_top_left not in list_visited:
                                list_cand.append(num_top_left)
                                list_visited.append(num_top_left)
                            else:
                                new_partition_num = (int(num_top / width) + 1) * width - 1
                                if new_partition_num not in list_visited:
                                    list_cand.append(new_partition_num)
                                    list_visited.append(new_partition_num)

                        num_top_right = num_center - width + 1
                        if num_top_right >= 0:
                            if num_top_right < (int(num_top / width) + 1) * width and num_top_right not in list_visited:
                                list_cand.append(num_top_right)
                                list_visited.append(num_top_right)
                            else:
                                new_partition_num = int(num_top / width) * width
                                if new_partition_num not in list_visited:
                                    list_cand.append(new_partition_num)
                                    list_visited.append(new_partition_num)

                        num_bottom_left = num_center + width - 1
                        if num_bottom_left < width * height:
                            if num_bottom_left <= int(num_bottom / width) * width and num_bottom_left not in list_visited:
                                list_cand.append(num_bottom_left)
                                list_visited.append(num_bottom_left)
                            else:
                                new_partition_num = (int(num_bottom / width) + 1) * width - 1
                                if new_partition_num not in list_visited:
                                    list_cand.append(new_partition_num)
                                    list_visited.append(new_partition_num)

                        num_bottom_right = num_center + width + 1
                        if num_bottom_right < width * height:
                            if num_bottom_right >= 0:
                                if num_bottom_right < (int(num_bottom / width) + 1) * width and num_bottom_right not in list_visited:
                                    list_cand.append(num_bottom_right)
                                    list_visited.append(num_bottom_right)
                                else:
                                    new_partition_num = int(num_bottom / width) * width
                                    if new_partition_num not in list_visited:
                                        list_cand.append(new_partition_num)
                                        list_visited.append(new_partition_num)

                        for cand in list_cand:
                            mmap_partition_num = str(int(cand))
                            loop_count = mmap_partition_num

                            filename = 'Tile-' + mmap_partition_num

                            lr = data_dic_lr[filename]
                            hr = data_dic_hr[filename]
                            lr, hr = self.prepare(lr, hr)
                            sr = self.model(lr, idx_scale)
                            sr = utility.quantize(sr, self.args.rgb_range)

                            save_list = [sr]
                            self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range, dataset=d
                            )
                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                self.ckp.save_results(d, filename[0], save_list, scale)

                            postfix = ('SR', 'LR', 'HR')
                            for v, p in zip(save_list, postfix):
                                normalized = v[0].mul(255 / self.args.rgb_range)
                                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

                                data_list = tensor_cpu.numpy().tolist()
                                new_list = []
                                get_list_num(data_list, new_list)

                                partitionBytesPosition = d_gap * d_gap * 3 * int(mmap_partition_num)
                                mmap_file.seek(partitionBytesPosition)

                                mmap_file.write(bytes(new_list))

                                # sr_list = []
                                # sr_list.append(int(mmap_partition_num) + 1)
                                # sr_mmap_file.write(bytes(sr_list))

                                sr_list = list(map(int, list(str(int(mmap_partition_num) + 1).zfill(4))))
                                sr_mmap_file.write(bytes(sr_list))

                                print("Partition " + mmap_partition_num)

                            self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                            best = self.ckp.log.max(0)
                            self.ckp.write_log(
                                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                    d.dataset.name,
                                    scale,
                                    self.ckp.log[-1, idx_data, idx_scale],
                                    best[0][idx_data, idx_scale],
                                    best[1][idx_data, idx_scale] + 1
                                )
                            )

        time.sleep(60)
        mmap_file.close()
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        # if self.args.save_results:
        #    self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
