import pandas as pd
import numpy as np
import yaml
import argparse

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def select_features(df):  # 提取 V1 + V2 特征
    v1_features = [
        'I1', 'I2', 'I3',
        'gx', 'gy', 'gz',
        'ax', 'ay', 'az',
        'V1real', 'V2real', 'V3real',
        'N1', 'N2', 'N3'
    ]
    
    selected_features = v1_features
    return df[selected_features]

def generate_binary_target(df):  # 将 Type==1 设为 1，其他设为 0
    return (df['Type'] == 4).astype(int)

def process_and_save(df, config, output_path):
    # 清理列名空格（若有）
    df.columns = df.columns.str.strip()
    
    X = select_features(df)
    y = generate_binary_target(df)

    result_df = X.copy()
    result_df['Type'] = y.values
    result_df.to_csv(output_path, index=False)

def main(config_path):
    config = load_config(config_path)

    # 读取训练数据
    train_df = pd.read_excel(config['data_load']['train_dataset']).dropna()

    # 处理训练数据（不归一化）
    process_and_save(
        train_df,
        config,
        config['data_split']['trainset_path']
    )

    # 读取测试数据
    test_df = pd.read_excel(config['data_load']['test_dataset']).dropna()

    # 处理测试数据（不归一化）
    process_and_save(
        test_df,
        config,
        config['data_split']['testset_path']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
