import numpy as np
import scipy.sparse as sp
import torch as th
from pathlib import Path
import re
from sklearn.preprocessing import OneHotEncoder
 
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    rowsum = np.array(features.sum(1), dtype=np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)




def _load_bipartite_edge_list(file_path, src_size, dst_size):
    file_path = Path(file_path)
    edges = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x for x in re.split(r"[,\s]+", line) if x]
            if len(parts) < 2:
                raise ValueError(f"{file_path}:{line_no}: expected at least two integer columns")
            try:
                src = int(parts[0])
                dst = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid integer edge: {line!r}") from exc
            edges.append((src, dst))

    if not edges:
        indices = th.empty((2, 0), dtype=th.long)
        values = th.empty((0,), dtype=th.float32)
        return th.sparse_coo_tensor(indices, values, (src_size, dst_size)).coalesce()

    src_ids = np.asarray([e[0] for e in edges], dtype=np.int64)
    dst_ids = np.asarray([e[1] for e in edges], dtype=np.int64)

    # Convert only when the data gives unambiguous evidence of one-based IDs.
    if src_ids.max() == src_size or dst_ids.max() == dst_size:
        src_ids = src_ids - 1
        dst_ids = dst_ids - 1

    if src_ids.min() < 0 or src_ids.max() >= src_size:
        raise ValueError(
            f"{file_path}: source ids out of range [0, {src_size - 1}] "
            f"(observed {src_ids.min()}..{src_ids.max()})"
        )
    if dst_ids.min() < 0 or dst_ids.max() >= dst_size:
        raise ValueError(
            f"{file_path}: destination ids out of range [0, {dst_size - 1}] "
            f"(observed {dst_ids.min()}..{dst_ids.max()})"
        )

    indices = th.from_numpy(np.vstack([src_ids, dst_ids]))
    values = th.ones(indices.shape[1], dtype=th.float32)
    return th.sparse_coo_tensor(indices, values, (src_size, dst_size)).coalesce()



def _as_scipy_support(matrix):
    if th.is_tensor(matrix):
        if matrix.layout == th.sparse_coo:
            matrix = matrix.coalesce()
            idx = matrix.indices().cpu().numpy()
            data = np.ones(idx.shape[1], dtype=np.int8)
            out = sp.coo_matrix((data, (idx[0], idx[1])), shape=tuple(matrix.shape)).tocsr()
        else:
            arr = matrix.detach().cpu().numpy()
            out = sp.csr_matrix(arr != 0, dtype=np.int8)
    else:
        out = sp.csr_matrix(matrix)
        out = out.copy()
        out.data = np.ones_like(out.data, dtype=np.int8)
        out.eliminate_zeros()
    return out


def _assert_same_support(expected, observed, name):
    expected = _as_scipy_support(expected)
    observed = _as_scipy_support(observed)
    if expected.shape != observed.shape:
        raise ValueError(
            f"DBLP relation validation failed for {name}: expected shape "
            f"{expected.shape}, observed {observed.shape}"
        )
    diff = (expected != observed).nnz
    if diff != 0:
        raise ValueError(
            f"DBLP relation validation failed for {name}: {diff} support entries differ. "
            "Check the orientation/indexing of pa.txt, pc.txt, and pt.txt before running experiments."
        )


def _validate_dblp_primitive_relations(pa, pc, pt, nei_ap, nei_apc, nei_apt):
    pa_s = _as_scipy_support(pa)
    pc_s = _as_scipy_support(pc)
    pt_s = _as_scipy_support(pt)
    expected_ap = pa_s.T.tocsr()
    expected_apc = (expected_ap @ pc_s).tocsr()
    expected_apt = (expected_ap @ pt_s).tocsr()
    expected_apc.data = np.ones_like(expected_apc.data, dtype=np.int8)
    expected_apt.data = np.ones_like(expected_apt.data, dtype=np.int8)
    _assert_same_support(expected_ap, nei_ap, "AP")
    _assert_same_support(expected_apc, nei_apc, "APC")
    _assert_same_support(expected_apt, nei_apt, "APT")


def load_acm(ratio, type_num):
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)#类别标签
    
    nei_a  = sp.load_npz(path + "nei_a.npz")#临街矩阵
    nei_s = sp.load_npz(path + "nei_s.npz")
    feat_p = sp.load_npz(path + "p_feat.npz")#特征矩阵
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")#p到a的关系矩阵
    psp = sp.load_npz(path + "psp.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]#验证样本索引

    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_s = sparse_mx_to_torch_sparse_tensor(nei_s)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, train, val, test


def load_dblp(ratio, type_num):
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)#节点类别标签
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")#
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_c = sp.eye(type_num[3])
    feat_t = np.load(path+"t_feat.npz")

    nei_ap = sp.load_npz(path + "nei_ap.npz")
    nei_apc = sp.load_npz(path + "nei_apc.npz")
    nei_apcp = sp.load_npz(path + "nei_apcp.npz")
    nei_apt = sp.load_npz(path + "nei_apt.npz")
    nei_aptp = sp.load_npz(path + "nei_aptp.npz")

    num_a, num_p, num_t, num_c = type_num
    path_relations = {
        "PA": _load_bipartite_edge_list(path + "pa.txt", num_p, num_a),
        "PC": _load_bipartite_edge_list(path + "pc.txt", num_p, num_c),
        "PT": _load_bipartite_edge_list(path + "pt.txt", num_p, num_t),
    }
    _validate_dblp_primitive_relations(
        path_relations["PA"], path_relations["PC"], path_relations["PT"],
        nei_ap, nei_apc, nei_apt,
    )

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz") 

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    nei_ap = sparse_mx_to_torch_sparse_tensor(nei_ap)
    nei_apc = sparse_mx_to_torch_sparse_tensor(nei_apc)
    nei_apcp = sparse_mx_to_torch_sparse_tensor(nei_apcp)
    nei_apt = sparse_mx_to_torch_sparse_tensor(nei_apt)
    nei_aptp = sparse_mx_to_torch_sparse_tensor(nei_aptp)
        
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_t = th.FloatTensor(feat_t)
    feat_c = th.FloatTensor(preprocess_features(feat_c))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_ap, nei_apc, nei_apcp, nei_apt, nei_aptp], [feat_a, feat_p, feat_t, feat_c], [apa, apcpa, aptpa], label, train, val, test, path_relations


def load_aminer(ratio, type_num):
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = sp.load_npz(path + "nei_a.npz")
    nei_r = sp.load_npz(path+ "nei_r.npz")
 
    feat_p_pap = np.load(path + "feat_p_pap.w1000.l100.npy").astype('float')
    feat_p_prp = np.load(path + "feat_p_prp.w1000.l100.npy").astype('float')
    feat_a = np.load(path + "feat_a.w1000.l100.npy").astype('float')
    feat_r = np.load(path + "feat_r.w1000.l100.npy").astype('float')

    feat_p = th.stack((th.FloatTensor(feat_p_pap),th.FloatTensor(feat_p_prp)))
    feat_a = th.FloatTensor(feat_a)
    feat_r = th.FloatTensor(feat_r)

    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_r = sparse_mx_to_torch_sparse_tensor(nei_r)

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], label, train, val, test


def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    return data






