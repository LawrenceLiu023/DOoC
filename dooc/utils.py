import typing
import networkx as nx
from collections import defaultdict
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


def load_gene_mapping(file_path: str) -> dict:
    res = {}

    with open(file_path) as f:
        for line in f:
            line = line.rstrip().split()
            res[line[1]] = int(line[0])

    return res


def load_ontology(file_name: str, gene2id_mapping: dict) -> typing.Sequence:
    dg = nx.DiGraph()
    term_direct_gene_map = defaultdict(set)

    term_size_map, gene_set = {}, set()

    file_handle = open(file_name)
    for line in file_handle:
        line = line.rstrip().split()
        if line[2] == "default":
            dg.add_edge(line[0], line[1])
            continue

        if line[1] not in gene2id_mapping:
            continue
        if line[0] not in term_direct_gene_map:
            term_direct_gene_map[line[0]] = set()

        term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])
        gene_set.add(line[1])
    file_handle.close()

    print("There are", len(gene_set), "genes")

    leaves = []
    for term in dg.nodes():
        term_gene_set = set()
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dg, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        if len(term_gene_set) == 0:
            raise ValueError(f"There is empty terms, please delete term: {term}")

        term_size_map[term] = len(term_gene_set)

        if dg.in_degree(term) == 0:
            leaves.append(term)

    ug = dg.to_undirected()
    connected_subg_list = list(nxacc.connected_components(ug))

    print("There are", len(leaves), "roots:", leaves[0])
    print("There are", len(dg.nodes()), "terms")
    print("There are", len(connected_subg_list), "connected componenets")

    if len(leaves) > 1:
        raise ValueError(
            "There are more than 1 root of ontology. Please use only one root."
        )

    if len(connected_subg_list) > 1:
        raise ValueError(
            "There are more than connected components. Please connect them."
        )

    return dg, leaves[0], term_size_map, term_direct_gene_map


class _ComparableItem:
    """
    将对象转化为可比较对象，从而对于列表，可以使用`sort()`进行排序。

    Attributes
    ----------
    item : Any
        需要添加比较功能的项，可以是任何类型。
    compare_func : Callable
        一个二元比较函数，大小比较功能基于这一函数实现。
    lt_return : dict, default -1
        当`compare_func`比较结果为小于时，返回的值。

    Methods
    -------
    __lt__(other)
        实现`<`比较运算符，python原生sort()依赖`<`运算符进行比较。如果不实现`<`运算符，`sort()`也可以支持`>`运算符。
    """

    def __init__(self, item, compare_func, lt_return=-1) -> None:
        self.item = item
        self.compare_func = compare_func
        self.lt_return = lt_return

    def __lt__(self, other) -> bool:
        """实现`<`比较运算符，python原生sort()依赖`<`运算符进行比较"""
        preference = self.compare_func(self.item, other.item)
        if preference == self.lt_return:
            return True
        return False


def pairwise_rank(
    items: typing.Sequence,
    compare_func: typing.Callable[[typing.Any, typing.Any], typing.Any],
    lt_return: typing.Any = -1,
) -> typing.Sequence[int]:
    """
    利用Pairwise推理的函数，得到全局排序结果。

    Parameters
    ----------
    items : Sequence
        需要比较的项。
    compare_func : Callable
        一个二元比较函数，返回一个表示大小比较结果的值。形式类似于`pair_compare(a, b) -> int`
    lt_return : Any, default -1
        `compare_func`比较结果为小于时，返回的值。

    Returns
    -------
    rank_list : Sequence[int]
        与输入的项`items`对应的排序编号列表。

    Examples
    --------
    >>> items = ["1", "7", "2", "3", "0"]
    >>> compare_func = lambda x, y: -1 if float(x) < float(y) else 1
    >>> pairwise_rank(items, pairwise_infer)
    [2, 5, 3, 4, 1]
    """
    comparable_items: list[_ComparableItem] = [
        _ComparableItem(item, compare_func, lt_return) for item in items
    ]
    comparable_items_index_zip_list: list = list(
        zip(comparable_items, list(range(len(comparable_items))))
    )
    comparable_items_index_zip_list.sort(key=lambda x: x[0])
    rank_list: list = [0 for _ in range(len(comparable_items))]
    for i, (_, original_index) in enumerate(comparable_items_index_zip_list):
        rank_list[original_index] = i + 1
    return rank_list
