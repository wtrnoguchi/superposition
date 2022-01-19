import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


class RunnerBase(object):
    def __init__(self):
        pass


class PredictionRunnerBase(RunnerBase):
    def gen_mask_prob(self, t, config):
        raise NotImplementedError

    def calc_loss(self, model, loss, pred, truth, epoch, t, config):
        _loss_self_vision = getattr(F, config.train.loss_function)(
            pred['self_vision'], truth['self_vision'])
        loss['self_vision'] += _loss_self_vision

    def run_seq(self, model, seq_loader, epoch, config, collect_data=False):
        loss = defaultdict(lambda: 0)
        data = []

        for x, truth, t, in seq_loader:

            if t == 0:
                model.init_state(x['self_vision'].shape[0])

            p_mask = self.gen_mask_prob(t, config)
            pred = model(
                x,
                p_mask_vision_self=p_mask,
                p_mask_vision_other=p_mask,
            )

            if collect_data:
                model_state = model.get_state()
                _data = {}

                data_itp = {'input': x, 'prediction': pred, 'truth': truth}
                for modal in [
                        'self_vision', 'self_motion', 'self_position',
                        'other_motion', 'other_position'
                ]:
                    _data[modal] = {
                        itp: val[modal]
                        for itp, val in data_itp.items()
                        if modal in data_itp[itp].keys()
                    }
                _data['state'] = model_state

                data.append((t, _data))

            self.calc_loss(model, loss, pred, truth, epoch, t, config)

        return loss, data
    
    def data_iterator(self, epoch, data_loader):
        pbar = tqdm(
            data_loader.load_sequence(),
            desc="epoch: {:d}".format(epoch),
        )
        return pbar

    def eval_epoch(
            self,
            model,
            epoch,
            config,
            data_loader,
            logger,
            saver=None,
    ):

        collect_data = saver is not None
        if collect_data:
            n_data, n_steps = data_loader.num_sequence()
            saver.set_num(n_data, n_steps)

        data_iter = self.data_iterator(epoch, data_loader)
        for seq_loader, batch_index in data_iter:
            loss, data = self.run_seq(
                model,
                seq_loader,
                epoch,
                config,
                collect_data,
            )

            if saver is not None:
                for t, d in data:
                    saver.add(epoch, batch_index, t, d)
                saver.do_save()

            logger.add(loss)

        logger.write_all(epoch)

    def train_epoch(
            self,
            model,
            optimizer,
            epoch,
            config,
            data_loader,
            logger,
    ):

        data_iter = self.data_iterator(epoch, data_loader)
        for seq_loader, batch_index in data_iter:
            loss, _ = self.run_seq(model, seq_loader, epoch, config)

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            logger.add(loss)

        logger.write_all(epoch)


class PredictionRunner(PredictionRunnerBase):
    def gen_mask_prob(self, t, config):

        return 0 if t == 0 else config.p_mask_vision


class PolicyPredictionRunner(PredictionRunnerBase):
    def gen_mask_prob(self, t, config):

        return 0 if t == 0 else 1


class PredictionFeaturePredictionRunner(PredictionRunner):
    def calc_loss(self, model, loss, pred, truth, epoch, t, config):
        super().calc_loss(model, loss, pred, truth, epoch, t, config)

        sv_enc = model.self_vision_encoder_module(truth['self_vision'])
        ov_enc = model.other_vision_encoder_module(truth['self_vision'])
        state = model.get_state()
        ss = state['self'][0]
        os = state['other'][0]
        pred_self_feature = model.predict_feature(ss.detach())
        pred_other_feature = model.predict_feature(os.detach())
        _loss_self = F.mse_loss(pred_self_feature, sv_enc.detach())
        _loss_other = F.mse_loss(pred_other_feature, ov_enc.detach())
        loss['feature_prediction_self'] += _loss_self
        loss['feature_prediction_other'] += _loss_other


class PolicyPredictionFeaturePredictionRunner(PolicyPredictionRunner):
    def calc_loss(self, model, loss, pred, truth, epoch, t, config):
        super().calc_loss(model, loss, pred, truth, epoch, t, config)

        sv_enc = model.self_vision_encoder_module(truth['self_vision'])
        ov_enc = model.other_vision_encoder_module(truth['self_vision'])
        state = model.get_state()
        ss = state['self'][0]
        os = state['other'][0]
        pred_self_feature = model.predict_feature(ss)
        pred_other_feature = model.predict_feature(os)
        _loss_self = F.mse_loss(pred_self_feature, sv_enc.detach())
        _loss_other = F.mse_loss(pred_other_feature, ov_enc.detach())
        loss['feature_prediction_self'] += _loss_self
        loss['feature_prediction_other'] += _loss_other


class AutoencoderRunner(RunnerBase):
    def data_iterator(self, epoch, data_loader):
        pbar = tqdm(
            data_loader.load_flatten(),
            desc="epoch: {:d}".format(epoch),
        )
        return pbar


    def eval_epoch(
            self,
            model,
            epoch,
            config,
            data_loader,
            logger,
            saver=None,
    ):
        collect_data = saver is not None
        if collect_data:
            n_data, n_steps = data_loader.num_flatten()
            saver.set_num(n_data, n_steps)

        data_iter = self.data_iterator(epoch, data_loader)
        for batch, batch_index in data_iter:
            loss = {'self_vision': 0}

            rec = model(batch['self_vision'], decode_from_other=collect_data)

            loss['self_vision'] += getattr(F, config.train.loss_function)(
                rec['self_vision'], batch['self_vision'])
            if collect_data:
                _data = {}

                data_itp = {'input': batch, 'reconstruction': rec}
                for modal in [
                        'self_vision', 'self_motion', 'self_position',
                        'other_vision', 'other_motion', 'other_position'
                ]:
                    _data[modal] = {
                        itp: val[modal]
                        for itp, val in data_itp.items()
                        if modal in data_itp[itp].keys()
                    }

                saver.add(epoch, batch_index[0], batch_index[1], _data)
                saver.do_save()

            logger.add(loss)

        logger.write_all(epoch)

    def train_epoch(
            self,
            model,
            optimizer,
            epoch,
            config,
            data_loader,
            logger,
    ):

        data_iter = self.data_iterator(epoch, data_loader)
        for batch, batch_index in data_iter:
            loss = {'self_vision': 0}

            rec = model(batch['self_vision'])

            loss['self_vision'] += getattr(F, config.train.loss_function)(
                rec['self_vision'], batch['self_vision'])

            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()

            logger.add(loss)

        logger.write_all(epoch)
