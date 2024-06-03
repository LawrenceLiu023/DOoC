import typing
import torch
from moltx import tokenizers
from dooc import models
from dooc.utils import pairwise_rank


class MutSmi:
    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        model: models.MutSmi,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.smi_tokenizer = smi_tokenizer

        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self.model = model

        self.device = device

    def _model_args(
        self, gene: typing.Sequence[int], smiles: str
    ) -> typing.Tuple[torch.Tensor]:
        smi_src = self.gen_smi_token(smiles)
        smi_tgt = self.gen_smi_token(self.smi_tokenizer.BOS + smiles + self.smi_tokenizer.EOS)
        gene_src = self.gen_gene_token(gene)
        return smi_src, smi_tgt, gene_src

    def gen_smi_token(self, smiles: str) -> torch.Tensor:
        tokens = self.smi_tokenizer(smiles)
        res = torch.zeros(len(tokens), dtype=torch.int)
        for i, tk in enumerate(tokens):
            res[i] = tk
        return res.to(self.device)

    def gen_gene_token(self, gene: typing.Sequence[float]) -> torch.Tensor:
        return torch.tensor(gene, dtype=torch.float).to(self.device)


class MutSmiXAttention(MutSmi):
    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        model: models.MutSmiXAttention,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(smi_tokenizer, model, device)

    def __call__(self, gene: typing.Sequence[int], smiles: str) -> float:
        smi_src, smi_tgt, gene_src = self._model_args(gene, smiles)
        pred = self.model(smi_src, smi_tgt, gene_src)
        return pred.item()


class MutSmiFullConnection(MutSmi):
    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        model: models.MutSmiFullConnection,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(smi_tokenizer, model, device)

    def __call__(self, gene: typing.Sequence[int], smiles: str) -> float:
        smi_src, smi_tgt, gene_src = self._model_args(gene, smiles)
        pred = self.model(smi_src, smi_tgt, gene_src)
        return pred.item()


class MutSmiPairwiseLtr(MutSmi):
    def __init__(
        self,
        smi_tokenizer: tokenizers.MoltxTokenizer,
        model: models.MutSmiFullConnection,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(smi_tokenizer, model, device)

    def __call__(
        self, gene: typing.Sequence[int], smiles: typing.Sequence[str]
    ) -> typing.Sequence[float]:
        smi_src, gene_src = self._model_args(gene, smiles)

        def pairwise_infer(a, b) -> int:
            """
            这是一个Pairwise推理的示例，模仿了Pairwise推理的函数形式，可根据两项给出代表大小关系的值。
            """
            VALID_PREFERENCE:dict[str, int] = {"gt": 1, "eq": 0, "lt": -1}
            if a > b:
                return VALID_PREFERENCE["gt"]
            elif a == b:
                return VALID_PREFERENCE["eq"]
            else:
                return VALID_PREFERENCE["lt"]

        # 根据Pairwise推理，计算排序结果。三个参数分别为：需排序项的列表、Pairwise推理函数、比较结果为“小于”时推理函数的返回值
        pred: typing.Sequence[int] = pairwise_rank(smi_src, pairwise_infer, -1)
        return pred
