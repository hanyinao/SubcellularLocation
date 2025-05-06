from sklearn.preprocessing import StandardScaler
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
import numpy as np
from propy import PyPro
import os

# 1、PseAAC特征提取函数


def calculate_pseaac(sequence, lambda_value):
    try:
        protein = PyPro.GetProDes(sequence)
        paac = protein.GetPAAC(lamda=lambda_value)
        return paac
    except Exception as e:
        print(f"Error for sequence {sequence[:10]}...: {e}")
        return None

# 2、EBGW特征提取函数


def calculate_ebgw_features(protein_seq, window_num):
    # 定义氨基酸分组
    neutral_non_polar = {'G', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W'}
    neutral_polar = {'Q', 'N', 'S', 'T', 'Y', 'C'}
    acidic = {'D', 'E'}
    basic = {'H', 'K', 'R'}
    group_combinations = [(neutral_non_polar | neutral_polar, acidic | basic),
                          (neutral_non_polar | acidic, neutral_polar | basic),
                          (neutral_non_polar | basic, neutral_polar | acidic)]

    ebgw_features = []
    for combination in group_combinations:
        group1, group2 = combination
        binary_seq = [1 if aa in group1 else 0 for aa in protein_seq]
        n = len(binary_seq)
        if n == 0:  # 处理空序列
            ebgw_features.extend([0.0] * window_num)
            continue
        window_size = max(1, n // window_num)  # 确保窗口大小至少为 1
        sub_seq_weights = []
        for i in range(0, n, window_size):
            sub_seq = binary_seq[i:i + window_size]
            sub_seq_weight = np.mean(sub_seq) if sub_seq else 0.0
            sub_seq_weights.append(sub_seq_weight)
        # 补全特征数量
        if len(sub_seq_weights) < window_num:
            sub_seq_weights += [0.0] * (window_num - len(sub_seq_weights))
        ebgw_features.extend(sub_seq_weights)
    return ebgw_features

# 3、psepssm特征提取函数


def parse_pssm(pssm_file):
    pssm_matrix = []
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:
            if line.strip() == "":
                continue
            values = line.split()[2:22]
            try:
                values = list(map(float, values))
                # 确保每个子列表长度为20，不足则零填充
                if len(values) < 20:
                    values.extend([0] * (20 - len(values)))
                pssm_matrix.append(values)
            except ValueError:
                print(f"Error processing {pssm_file}: {line}")
    if not pssm_matrix:
        return np.array([])
    pssm_matrix = np.array(pssm_matrix)
    # 标准化处理
    means = np.mean(pssm_matrix, axis=0)
    stds = np.std(pssm_matrix, axis=0)
    stds[stds == 0] = 1e-8  # 避免除零错误
    pssm_matrix = (pssm_matrix - means) / stds
    return pssm_matrix


def calculate_psepssm(pssm_matrix, xi_max, w):
    if pssm_matrix.size == 0:
        return np.array([])
    L, m = pssm_matrix.shape
    avg_pssm = np.mean(pssm_matrix, axis=0)
    g_features = []
    for xi in range(1, xi_max + 1):
        if L <= xi:
            continue
        for j in range(m):
            diff_sq = np.sum((pssm_matrix[:-xi, j] - pssm_matrix[xi:, j]) ** 2)
            g = diff_sq / (L - xi)
            g_features.append(g)

    return np.concatenate([avg_pssm * (1 - w), np.array(g_features) * w])

# 4、标签共线性特征提取


def calculate_G(labels):
    # 计算标签共现矩阵
    label_cooccurrence = np.dot(labels.T, labels)

    # 构建标签图
    G = nx.Graph()
    label_names = labels.columns
    for i in range(len(label_names)):
        for j in range(i + 1, len(label_names)):
            denominator = label_cooccurrence[i, i] + \
                label_cooccurrence[j, j] - label_cooccurrence[i, j]
            if denominator == 0:
                sim = 0  # 处理除零错误
            else:
                sim = label_cooccurrence[i, j] / denominator
            G.add_edge(label_names[i], label_names[j], weight=sim)
    return G


def calculate_node2vec(G, labels, dimensions, walk_length, num_walks, window):
    if G.number_of_edges() == 0:
        raise ValueError("标签图没有有效边，无法生成嵌入")
    label_names = labels.columns
    # Node2Vec 嵌入
    try:
        node2vec = Node2Vec(G, dimensions=dimensions,
                            walk_length=walk_length, num_walks=num_walks)
        model = node2vec.fit(window=window)
        label_embeddings = pd.DataFrame(
            [model.wv[node] for node in label_names], index=label_names)
        return label_embeddings.T
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def generate_all(df, labels, G, params):
    """
    生成所有特征
    :param df: 数据框
    :param labels: 标签列表
    :param G: 标签图
    :param params: 参数字典
    :return: 特征矩阵
    """
    # 1、计算PseAAC特征
    lambda_value = params['pseaac_lambda']
    df["pseaac"] = df["Sequence"].apply(
        lambda x: calculate_pseaac(x, lambda_value=lambda_value))
    df_pseaac = pd.json_normalize(df["pseaac"])
    df = pd.concat([df, df_pseaac.add_prefix("pseaac_")], axis=1)
    df.drop(columns='pseaac', inplace=True)

    # 2、计算EBGW特征
    window_num = params['ebgw_window_size']
    df['EBGW_Features'] = df['Sequence'].apply(
        lambda x: calculate_ebgw_features(x, window_num=window_num))
    features_df = pd.DataFrame(df['EBGW_Features'].tolist())
    feature_columns = [f'EBGW_{i}' for i in range(features_df.shape[1])]
    features_df.columns = feature_columns
    df = pd.concat([df.drop('EBGW_Features', axis=1), features_df], axis=1)

    # 3、计算PsePSSM特征
    pssm_dir = r"D:\Han\software\Math\CSUpan\ShareCache\尹涵(数学与统计学院)\赛\一流专业人才计划\Codes\pssm_sequences"
    all_features = {}
    xi_max = params['psepssm_lambda']
    w = params['psepssm_w']
    for filename in os.listdir(pssm_dir):
        if filename.endswith('.pssm'):
            seq_name = filename.split('.')[0]
            pssm = parse_pssm(os.path.join(pssm_dir, filename))
            if pssm.size == 0:
                continue
            features = calculate_psepssm(pssm, xi_max=xi_max, w=w)
            all_features[seq_name] = features
    header = ['avg_{}'.format(i + 1) for i in range(20)]
    for xi in range(1, xi_max + 1):
        header += ['g_{}_{}'.format(xi, j + 1) for j in range(20)]
    rows_to_drop = []
    for index, row in df.iterrows():
        entry_name = row['Entry Name']
        if entry_name in all_features:
            features = all_features[entry_name]
            for i, col in enumerate(header):
                df.at[index, col] = features[i]
        else:
            print(
                f"Skipping entry {entry_name} as no corresponding PSSM file found.")
            rows_to_drop.append(index)
    df = df.drop(rows_to_drop)

    # 4、提取标签共现性特征
    dimensions = params['n2v_dimensions']
    walk_length = params['n2v_walk_length']
    num_walks = params['n2v_num_walks']
    window = params['n2v_window']
    label_embeddings = calculate_node2vec(
        G, labels, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, window=window)
    if label_embeddings is None:
        print("Failed to generate label embeddings. Returning None.")
        return None
    X_label_feat = []
    for labels in df['subcellular_location']:
        valid_labels = [lb for lb in labels if lb in label_embeddings]
        if len(valid_labels) > 0:
            feat = np.mean([label_embeddings[lb]
                           for lb in valid_labels], axis=0)
        else:
            feat = np.zeros(dimensions)
        X_label_feat.append(feat)
    X_label_feat = np.array(X_label_feat)
    n = X_label_feat.shape[1]
    label_feat_columns = [f'label_feat_{i}' for i in range(n)]
    X_label_feat_df = pd.DataFrame(X_label_feat, columns=label_feat_columns)

    # 5、标签拼接及标准化
    columns_to_drop = ['Entry', 'Entry Name', 'Sequence',
                       'subcellular_location', 'ec_number', 'pseaac']
    features_left = df.drop(columns=columns_to_drop)
    features_combined = pd.concat([features_left, X_label_feat_df], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_combined)

    return scaled_features
