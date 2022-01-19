import torch


def mask(x, p_mask):

    mask = torch.rand((x.size(0), 1), device=x.device) < p_mask
    return torch.where(mask, torch.zeros_like(x), x)


def same_mask(xs, p_mask):

    mask = torch.rand((xs[0].size(0), 1), device=xs[0].device) < p_mask
    return [torch.where(mask, torch.zeros_like(x), x) for x in xs]


def init_fc(fc, nonlinearity, method):

    if method == "xavier":
        torch.nn.init.xavier_normal_(
            fc.weight.data, torch.nn.init.calculate_gain(nonlinearity)
        )
    elif method == "he":
        torch.nn.init.kaiming_normal_(
            fc.weight.data, torch.nn.init.calculate_gain(nonlinearity)
        )
    if fc.bias is not None:
        fc.bias.data.fill_(0.0)


def init_conv(conv, nonlinearity, method):

    if method == "xavier":
        torch.nn.init.xavier_normal_(
            conv.weight.data, torch.nn.init.calculate_gain(nonlinearity)
        )
    elif method == "he":
        torch.nn.init.kaiming_normal_(
            conv.weight.data, torch.nn.init.calculate_gain(nonlinearity)
        )
    conv.bias.data.fill_(0.0)


def init_conv_transposed(conv, nonlinearity, method):

    if method == "xavier":
        torch.nn.init.xavier_normal_(
            conv.weight.data.transpose(0, 1), torch.nn.init.calculate_gain(nonlinearity)
        )
    elif method == "he":
        torch.nn.init.kaiming_normal_(
            conv.weight.data.transpose(0, 1), torch.nn.init.calculate_gain(nonlinearity)
        )

    conv.bias.data.fill_(0.0)


def init_lstm(lstm):

    wii, wif, wig, wio = torch.chunk(lstm.weight_ih.data, 4, 0)
    whi, whf, whg, who = torch.chunk(lstm.weight_hh.data, 4, 0)
    bii, bif, big, bio = torch.chunk(lstm.bias_ih.data, 4, 0)
    bhi, bhf, bhg, bho = torch.chunk(lstm.bias_hh.data, 4, 0)

    torch.nn.init.xavier_normal_(wii, torch.nn.init.calculate_gain("sigmoid"))
    torch.nn.init.xavier_normal_(wif, torch.nn.init.calculate_gain("sigmoid"))
    torch.nn.init.xavier_normal_(wig, torch.nn.init.calculate_gain("tanh"))
    torch.nn.init.xavier_normal_(wio, torch.nn.init.calculate_gain("sigmoid"))

    torch.nn.init.xavier_normal_(whi, torch.nn.init.calculate_gain("sigmoid"))
    torch.nn.init.xavier_normal_(whf, torch.nn.init.calculate_gain("sigmoid"))
    torch.nn.init.xavier_normal_(whg, torch.nn.init.calculate_gain("tanh"))
    torch.nn.init.xavier_normal_(who, torch.nn.init.calculate_gain("sigmoid"))

    bii.data.fill_(0.0)
    bif.data.fill_(1.0)
    big.data.fill_(0.0)
    bio.data.fill_(0.0)

    bhi.data.fill_(0.0)
    bhf.data.fill_(0.0)
    bhg.data.fill_(0.0)
    bho.data.fill_(0.0)
