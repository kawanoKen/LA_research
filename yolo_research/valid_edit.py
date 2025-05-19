import os

def edit_and_remove_labels(valid_dir):
    """
    `valid`ディレクトリ内のラベルファイルを編集し、ラベルが1のものを0に、3のものを1に変更します。
    また、ラベルが1または3で始まる画像ファイルとラベルファイルを削除します。

    Parameters:
    valid_dir (str): validディレクトリのパス
    """
    # validディレクトリ内のファイルを取得
    image_dir = os.path.join(valid_dir, 'images')
    label_dir = os.path.join(valid_dir, 'labels')
    
    # 画像とラベルファイルを確認
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    
    # すべてのラベルファイルをチェック
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        # ラベルファイルを読み込む
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # ラベルファイル内のラベルを編集
        updated_lines = []
        for line in lines:
            parts = line.split()
            label = int(parts[0])

            # ラベルが1なら0に、ラベルが3なら1に変換
            if label == 1:
                parts[0] = '0'
            elif label == 3:
                parts[0] = '1'
            else:
                continue

            updated_lines.append(" ".join(parts) + "\n")

        # 編集した内容をファイルに書き込む
        with open(label_path, 'w') as f:
            f.writelines(updated_lines)


# 使用例
print("validディレクトリを編集します！！！気をつけてください！！！")
breakpoint()
valid_directory = "/work/kawano/LA/datasets/african-wildlife_noise/valid"  # validディレクトリのパスに変更
edit_and_remove_labels(valid_directory)