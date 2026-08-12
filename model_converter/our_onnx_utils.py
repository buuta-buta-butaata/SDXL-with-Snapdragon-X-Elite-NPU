import onnx
import os
import gc

def find_boundary(model, output_dir):
    graph = model.graph
     
    print("=== UNetの境界テンソルを探索中 ===")
     
    down_end_tensor = None
    mid_end_tensor = None
    up_0_mid_tensor = None
     
    # スキップコネクション（DownからUpへ渡されるもの）を格納するセット
    down_outputs = set()
    up_inputs = set()
     
    # 各ブロックの共通部分を抜き出すためのセット
    mid_inputs_from_down = set()
     
    for node in graph.node:
        # metadata_propsからname_scopesを取得
        scopes = ""
        for prop in node.metadata_props:
            if prop.key == "pkg.torch.onnx.name_scopes":
                scopes = prop.value
                break
                
        if not scopes:
            continue
     
        # down_blocksに含まれないスキップコネクション
        if "unet.conv_in" in scopes:
            for out in node.output:
                down_outputs.add(out)
            
        # up_blocksに含まれないスキップコネクション
        if "unet.conv_out" in scopes:
            for inp in node.input:
                up_inputs.add(inp)
     
        # 1. Downブロックの出力を監視
        if "unet.down_blocks" in scopes:
            # 定数を除外
            if node.op_type != "Constant":
                for out in node.output:
                    down_outputs.add(out)
     
        # 2. Upブロックの入力を監視
        if "unet.up_blocks" in scopes:
            # up_begin_inputs.extend(node.input)
            for inp in node.input:
                up_inputs.add(inp)
     
        # 3. midブロックの入力を監視
        if "unet.mid_block" in scopes:
            # 定数を除外
            if node.op_type != "Constant":
                for inp in node.input:
                    mid_inputs_from_down.add(inp)
                    
        # down_blocks.2 に属するノードの出力を常に上書き保存（結果的に最後のノードの出力が残る）
        if "unet.down_blocks.2" in scopes:
            if node.output:
                down_end_tensor = node.output[0]
     
        # mid_block に属するノードの出力を常に上書き保存（結果的に最後のノードの出力が残る）
        if "unet.mid_block" in scopes:
            if node.output:
                mid_end_tensor = node.output[0]
     
        # up_blocks.0 の中の 1つ目のアテンション/レズネット層（attentions.0）の出力を狙う
        if "unet.up_blocks.0.attentions.0" in scopes:
            if node.output:
                up_0_mid_tensor = node.output[0]
                
    # スキップコネクションの特定（Downの出力であり、かつUpの入力であるもの）
    skip_connections = down_outputs.intersection(up_inputs)
    common_outputs = sorted(skip_connections.intersection(mid_inputs_from_down))
     
    print(f"\n[1] Part1 (Down) の最終出力テンソル:")
    print(f"  -> {down_end_tensor}")
     
    print(f"\n[2] Part2 (Mid) の最終出力テンソル:")
    print(f"  -> {mid_end_tensor}")
     
    print(f"\n[3] Part3 (Up_1) の最終出力テンソル:")
    print(f"  -> {up_0_mid_tensor}")
     
    print(f"\n[4] 検出されたdown_blocksの出力で、mid, up_blocksへの共通入力 (Part2,3,4の追加入力として必要なテンソル群):")
    common_outputs.remove(down_end_tensor)
    for common in common_outputs:
        skip_connections.remove(common)
        print(f"  - {common}");
     
    print(f"\n[5] up_blocksへのスキップコネクション (Part4の追加入力として必要なテンソル群):")
    skip_connections = sorted(list(skip_connections))
    for skip in skip_connections:
        # テンソル名が自動生成のIDになっているのでそのまま表示
        print(f"  - {skip}")
        
    return down_end_tensor, mid_end_tensor, up_0_mid_tensor, skip_connections, common_outputs

def split_unet(model, input_model_path, output_dir, down_end_tensor, mid_end_tensor, up_0_mid_tensor, skip_connections, common_outputs):
    # 1. 元々のUNetの入力テンソル名
    #base_inputs = [i.name for i in model.graph.input]
    base_inputs = [i.name for i in model.graph.input]
    # encoder_hidden_statesだけは全部品の共通の入力になる 面倒なのでdimで判断しちゃう
    encoder_hidden_states = [i.name for i in model.graph.input if len(i.type.tensor_type.shape.dim) == 3]
    common_inputs = [i.name for i in model.graph.input if len(i.type.tensor_type.shape.dim) < 3]
    # 一応順番は保持しておく
    down_inputs = list(filter(lambda x: x not in common_inputs, base_inputs))
     
    print(f"\nbase inputs {base_inputs}")
    print(f"encoder_hidden_states {encoder_hidden_states}")
    print(f"common inputs {common_inputs}")
     
    # part3で処理済みの分は入力から除外する
    part4_inputs = skip_connections.copy()
    part4_inputs.remove(down_end_tensor)
    part4_inputs = [up_0_mid_tensor] + part4_inputs + common_outputs + encoder_hidden_states
     
    final_outputs = [o.name for o in model.graph.output]
     
    parts = [
        {
            "inputs": common_inputs,
            "outputs": common_outputs,
            "output_path": os.path.join(output_dir, "unet_part0.onnx"),
        },
        {
            "inputs": common_outputs + down_inputs,
            "outputs": skip_connections,
            "output_path": os.path.join(output_dir, "unet_part1.onnx"),
        },
        {
            "inputs": [down_end_tensor] + common_outputs + encoder_hidden_states,
            "outputs": [mid_end_tensor],
            "output_path": os.path.join(output_dir, "unet_part2.onnx"),
        },
        {
            "inputs": [down_end_tensor] + [mid_end_tensor] + common_outputs + encoder_hidden_states,
            "outputs": [up_0_mid_tensor],
            "output_path": os.path.join(output_dir, "unet_part3.onnx"),
        },
        {
            "inputs": part4_inputs,
            "outputs": final_outputs,
            "output_path": os.path.join(output_dir, "unet_part4.onnx"),
        }
    ]
     
    print("\n=== ONNXモデルの分割を開始します ===")
    
    for i,p in enumerate(parts):
        print(f"[{i}/4] Part{i} を抽出中...")
        onnx.utils.extract_model(
            input_model_path,
            p["output_path"],
            input_names=p["inputs"],
            output_names=p["outputs"],
        )
     
        gc.collect()
    
    print("=== 分割完了しました！ ===")
    
    print("\n=== input/outputと重複する分のvalue_infoを削除する処理 ===")
    for i,p in enumerate(parts):
        sanitize_value_info(p["output_path"])
        generate_specs(i, p["output_path"])
    

def sanitize_value_info(model_path):
    # モデルの読み込み
    model = onnx.load(model_path)
    graph = model.graph

    # 1. 現在の「入力（inputs）」と「出力（outputs）」の名前をすべて収集
    io_names = set()
    for inp in graph.input:
        io_names.add(inp.name)
    for out in graph.output:
        io_names.add(out.name)

    # 2. value_info を走査し、入出力に含まれている名前のものを除外する
    new_value_info = []
    removed_count = 0
    
    for info in graph.value_info:
        if info.name in io_names:
            # 入出力と重複している場合はスキップ（削除）
            removed_count += 1
            continue
        new_value_info.append(info)

    # 3. グラフの value_info をクレンジングされた新しいリストに置き換える
    del graph.value_info[:]
    graph.value_info.extend(new_value_info)

    # 保存
    onnx.save_model(
        model,
        model_path,
        # save_as_external_data=True, 
        # all_tensors_to_one_file=True,
    )
    print(f"\n[{model_path}] クレンジング完了: {removed_count} 個の重複した value_info を削除しました。")



def zatsu_type2str(inp):
    if inp == 1:
        return "float32"
    elif inp == 10:
        return "float16"
    return "unknown"

def generate_specs(part_num, path):
    print(f"\nFor {path} " + "=" * (75 - len(f"For {path} ")))
    model = onnx.load(path, load_external_data=False)
    graph = model.graph

    inputs = graph.input
    outputs = graph.output

    input_specs = []
    input_names = []
    for inp in inputs:
        name = inp.name
        type = inp.type.tensor_type
        dim = type.shape.dim

        input_names.append(f"\"{name}\"")
        if len(dim) == 1:
            # print(f"{name}=(({dim[0].dim_value}, ), \"{zatsu_type2str(type.elem_type)}\")")
            dim_values = f"({dim[0].dim_value}, )"
            # input_specs.append(f"{name}=(({dim[0].dim_value}, ), \"{zatsu_type2str(type.elem_type)}\")")
        else:
            # print(f"{name}=(({', '.join(map(lambda x: str(x.dim_value), dim))}), \"{zatsu_type2str(type.elem_type)}\")")
            dim_values = f"({', '.join(map(lambda x: str(x.dim_value), dim))})"
        # print(f"{name}=({dim_values}, \"{zatsu_type2str(type.elem_type)}\")")
        input_specs.append(f"{name}=({dim_values}, \"{zatsu_type2str(type.elem_type)}\")")

    output_names = []
    for out in outputs:
        output_names.append(out.name)

    specs = f"input_names = [{', '.join(input_names)}]"
    specs += f"\ninput_specs = dict({', '.join(input_specs)})"
    specs += f"\noutput_names = \"{','.join(output_names)}\""
    specs += f"\ncompile_options = \"--truncate_64bit_io --output_names {','.join(output_names)}\""
    print(specs)

    temp_dir = "./generated"
    os.makedirs(temp_dir, exist_ok=True)
    output_IO_specs(f"{temp_dir}/unet_part{part_num}_specs.py", specs)

def output_IO_specs(file_name, text):
    with open(file_name, "w") as f:
        f.write(text)
    

# print("\nプリコンパイル時に利用する入力情報、コンパイルオプションを出力")
# generate_specs(common_path)
# generate_specs(down_path)
# generate_specs(mid_path)
# generate_specs(up_1_path)
# generate_specs(up_2_path)

if __name__ == "__main__":
    exit()

