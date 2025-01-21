import os
import json
import numpy as np


def calculate_mae(mol1, mol2):
    """
    计算 mol1 和 mol2 对应原子坐标的 MAE (Mean Absolute Error)
    """
    coords1 = np.array([site['xyz'] for site in mol1['sites']])
    coords2 = np.array([site['xyz'] for site in mol2['sites']])
    if coords1.shape != coords2.shape:
        raise ValueError("The number of sites in mol1 and mol2 does not match.")
    mae = np.mean(np.abs(coords1 - coords2))
    return mae


def process_json_file(filepath):
    """
    处理一个 JSON 文件，计算每个样本的 MAE
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    maes = []
    for sample in data:
        try:
            mol1 = sample['mol1']
            mol2 = sample['mol2']
            mae = calculate_mae(mol1, mol2)
            maes.append(mae)
        except Exception as e:
            print(f"Error processing sample: {e}")
    return maes


def main(directory, output_file):
    """
    遍历文件夹中的所有 JSON 文件，计算 MAE 并保存到新的 JSON 文件中
    """
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            try:
                maes = process_json_file(filepath)
                results.append({"file": filename, "mae_results": maes})
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # 将结果保存到一个 JSON 文件中
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
    print(f"MAE results saved to {output_file}")


if __name__ == "__main__":
    # 替换为你的文件夹路径
    directory = r"/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol_1000/json"
    # 保存结果的 JSON 文件路径
    output_file = os.path.join(directory, "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol_1000/mae/mae.json")
    main(directory, output_file)
