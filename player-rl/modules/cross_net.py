import torch
from torch import nn


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model"""

    def __init__(
        self, in_features, layer_num=2, parameterization="vector",
    ):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == "vector":
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList(
                [
                    nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1)))
                    for i in range(self.layer_num)
                ]
            )
        elif self.parameterization == "matrix":
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList(
                [
                    nn.Parameter(
                        nn.init.xavier_normal_(torch.empty(in_features, in_features))
                    )
                    for i in range(self.layer_num)
                ]
            )
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.ParameterList(
            [
                nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1)))
                for _ in range(self.layer_num)
            ]
        )

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == "vector":
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == "matrix":
                dot_ = torch.matmul(
                    self.kernels[i], x_l
                )  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class CrossNetMix(nn.Module):
    """The Cross Network part of DCN-Mix model"""

    def __init__(
        self, in_features, low_rank=4, num_experts=4, layer_num=2, device="cpu"
    ):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, in_features, low_rank)
                    )
                )
                for _ in range(self.layer_num)
            ]
        )
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(num_experts, in_features, low_rank)
                    )
                )
                for i in range(self.layer_num)
            ]
        )
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))
                )
                for i in range(self.layer_num)
            ]
        )
        self.gating = nn.ModuleList(
            [nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)]
        )
        self.bias = torch.nn.ParameterList(
            [
                nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self.layer_num)
            ]
        )
        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(
                    self.V_list[i][expert_id].t(), x_l
                )  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(
                    self.U_list[i][expert_id], v_x
                )  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(
                output_of_experts, 2
            )  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(
                gating_score_of_experts, 1
            )  # (bs, num_experts, 1)
            moe_out = torch.matmul(
                output_of_experts, gating_score_of_experts.softmax(1)
            )
            x_l = moe_out + x_l  # (bs, in_features, 1)
        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
