"""
3D Front场景多模态数据生成Pipeline流程图
使用graphviz生成可视化流程图
"""

from graphviz import Digraph

def create_3dfront_pipeline():
    dot = Digraph(comment='3D Front多模态数据生成Pipeline', format='png')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 设置颜色方案
    color_input = '#E8F4F8'
    color_process = '#FFF4E6'
    color_output = '#E8F5E9'
    color_decision = '#FFE5E5'
    
    # 1. 输入阶段
    dot.node('A0', '3D Front场景数据库', fillcolor=color_input)
    
    # 2. 初始化阶段
    dot.node('A1', '选择初始场景', fillcolor=color_process)
    dot.node('A2', '加载场景元数据\n(房间布局、物体库)', fillcolor=color_process)
    
    # 3. 增量生成循环
    dot.node('B1', '初始化:\nstep = 0\n当前物体列表 = []', fillcolor=color_process, shape='parallelogram')
    
    # 决策节点
    dot.node('C1', '是否继续添加物体?', fillcolor=color_decision, shape='diamond')
    
    # 4. 物体添加阶段
    dot.node('D1', '选择下一个物体\n(基于添加策略)', fillcolor=color_process)
    dot.node('D2', '计算物体位置和姿态\n(碰撞检测、空间约束)', fillcolor=color_process)
    dot.node('D3', '将物体添加到场景\nstep += 1', fillcolor=color_process)
    
    # 5. 多模态数据生成
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='多模态数据生成', style='filled', color='lightgrey')
        c.node('E1', '1. 场景渲染', fillcolor='#BBDEFB')
        c.node('E2', '2. 点云提取', fillcolor='#BBDEFB')
        c.node('E3', '3. 文本描述生成', fillcolor='#BBDEFB')
        c.node('E4', '4. 元数据记录', fillcolor='#BBDEFB')
    
    # 6. 渲染子流程
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='渲染模块', style='filled', color='#E1F5FE')
        c.node('E1_1', '设置相机参数\n(位置、角度、FOV)', fillcolor='white')
        c.node('E1_2', '设置光照\n(环境光、方向光)', fillcolor='white')
        c.node('E1_3', '多视角渲染\n(正视图、俯视图等)', fillcolor='white')
        c.node('E1_4', '保存渲染图片\n(PNG/JPG)', fillcolor=color_output)
    
    # 7. 点云子流程
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='点云提取模块', style='filled', color='#F3E5F5')
        c.node('E2_1', '场景网格采样', fillcolor='white')
        c.node('E2_2', '归一化坐标', fillcolor='white')
        c.node('E2_3', '保存点云数据\n(PCD/NPY)', fillcolor=color_output)
    
    # 8. 文本生成子流程
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='文本生成模块', style='filled', color='#FFF9C4')
        c.node('E3_1', '生成累积描述\n"房间有A、B、C"', fillcolor='white')
        c.node('E3_2', '生成增量描述\n"添加了C在B旁边"', fillcolor='white')
        c.node('E3_3', '生成空间关系描述\n(相对位置、功能关系)', fillcolor='white')
        c.node('E3_4', '保存文本描述\n(JSON/TXT)', fillcolor=color_output)
    
    # 9. 元数据记录
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='元数据记录模块', style='filled', color='#E0F2F1')
        c.node('E4_1', '记录物体列表', fillcolor='white')
        c.node('E4_2', '记录位置和姿态', fillcolor='white')
        c.node('E4_3', '记录相机参数', fillcolor='white')
        c.node('E4_4', '保存元数据\n(JSON)', fillcolor=color_output)
    
    # 10. 数据保存
    dot.node('F1', '保存当前step数据\nscene_id/step_n/', fillcolor=color_output)
    
    # 11. 循环控制
    dot.node('G1', '达到最大物体数\n或场景已满', fillcolor=color_decision, shape='diamond')
    
    # 12. 最终输出
    dot.node('H1', '生成数据集索引文件\n(训练/验证/测试划分)', fillcolor=color_process)
    dot.node('H2', '数据集完成\n可用于训练', fillcolor=color_output, shape='box3d')
    
    # 连接节点
    dot.edge('A0', 'A1')
    dot.edge('A1', 'A2')
    dot.edge('A2', 'B1')
    dot.edge('B1', 'C1')
    
    # 增量循环
    dot.edge('C1', 'D1', label='是')
    dot.edge('C1', 'H1', label='否')
    dot.edge('D1', 'D2')
    dot.edge('D2', 'D3')
    dot.edge('D3', 'E1')
    
    # 多模态并行生成
    dot.edge('E1', 'E1_1')
    dot.edge('E1_1', 'E1_2')
    dot.edge('E1_2', 'E1_3')
    dot.edge('E1_3', 'E1_4')
    
    dot.edge('E1', 'E2', style='invis')  # 布局用
    dot.edge('E2', 'E2_1')
    dot.edge('E2_1', 'E2_2')
    dot.edge('E2_2', 'E2_3')
    
    dot.edge('E2', 'E3', style='invis')  # 布局用
    dot.edge('E3', 'E3_1')
    dot.edge('E3_1', 'E3_2')
    dot.edge('E3_2', 'E3_3')
    dot.edge('E3_3', 'E3_4')
    
    dot.edge('E3', 'E4', style='invis')  # 布局用
    dot.edge('E4', 'E4_1')
    dot.edge('E4_1', 'E4_2')
    dot.edge('E4_2', 'E4_3')
    dot.edge('E4_3', 'E4_4')
    
    # 汇聚到保存节点
    dot.edge('E1_4', 'F1')
    dot.edge('E2_3', 'F1')
    dot.edge('E3_4', 'F1')
    dot.edge('E4_4', 'F1')
    
    # 循环回去
    dot.edge('F1', 'G1')
    dot.edge('G1', 'C1', label='继续')
    dot.edge('G1', 'H1', label='完成')
    
    dot.edge('H1', 'H2')
    
    return dot


def create_simplified_pipeline():
    """创建简化版流程图"""
    dot = Digraph(comment='3D Front Pipeline简化版', format='png')
    dot.attr(rankdir='LR', size='16,8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    color_stage = ['#E3F2FD', '#F3E5F5', '#FFF9C4', '#E8F5E9', '#FFE0B2']
    
    dot.node('S1', '🗂️\n3D Front\n场景数据库', fillcolor=color_stage[0])
    dot.node('S2', '🔄\n增量添加物体\n(逐个添加)', fillcolor=color_stage[1])
    dot.node('S3', '🎨\n多模态生成\n(渲染+点云+文本)', fillcolor=color_stage[2])
    dot.node('S4', '💾\n保存step数据\n(scene/step_n/)', fillcolor=color_stage[3])
    dot.node('S5', '📦\n数据集\n(训练集)', fillcolor=color_stage[4])
    
    dot.edge('S1', 'S2', label='加载场景')
    dot.edge('S2', 'S3', label='每次添加后')
    dot.edge('S3', 'S4', label='生成多模态数据')
    dot.edge('S4', 'S2', label='继续添加', style='dashed')
    dot.edge('S4', 'S5', label='场景完成')
    
    return dot


def create_data_structure_diagram():
    """创建数据结构示意图"""
    dot = Digraph(comment='数据结构', format='png')
    dot.attr(rankdir='TB', size='10,12')
    dot.attr('node', shape='folder', style='filled', fontname='Courier New')
    dot.attr('edge', fontname='SimHei')
    
    dot.node('root', '📁 dataset_root', fillcolor='#FFECB3')
    
    # 场景级别
    dot.node('scene1', '📁 scene_0001', fillcolor='#E1BEE7')
    dot.node('scene2', '📁 scene_0002', fillcolor='#E1BEE7')
    dot.node('scene3', '📁 ...', fillcolor='#E1BEE7', shape='plaintext')
    
    # Step级别
    dot.node('step0', '📁 step_00 (空场景)', fillcolor='#C5E1A5')
    dot.node('step1', '📁 step_01 (1个物体)', fillcolor='#C5E1A5')
    dot.node('step2', '📁 step_02 (2个物体)', fillcolor='#C5E1A5')
    dot.node('stepn', '📁 ...', fillcolor='#C5E1A5', shape='plaintext')
    
    # 文件级别
    dot.node('files', '''📄 render.png
📄 render_top.png
📄 render_side.png
📄 pointcloud.pcd
📄 metadata.json
📄 text_full.txt
📄 text_incremental.txt
📄 camera_params.json''', fillcolor='#B3E5FC', shape='note', fontname='Courier New')
    
    dot.edge('root', 'scene1')
    dot.edge('root', 'scene2')
    dot.edge('root', 'scene3', style='invis')
    
    dot.edge('scene1', 'step0')
    dot.edge('scene1', 'step1')
    dot.edge('scene1', 'step2')
    dot.edge('scene1', 'stepn', style='invis')
    
    dot.edge('step1', 'files')
    
    return dot


if __name__ == '__main__':
    print("生成3D Front多模态数据Pipeline流程图...")
    
    # 1. 详细流程图
    dot_detailed = create_3dfront_pipeline()
    dot_detailed.render('/workspace/pipeline_detailed', view=False, cleanup=True)
    print("✓ 详细流程图已生成: pipeline_detailed.png")
    
    # 2. 简化流程图
    dot_simple = create_simplified_pipeline()
    dot_simple.render('/workspace/pipeline_simplified', view=False, cleanup=True)
    print("✓ 简化流程图已生成: pipeline_simplified.png")
    
    # 3. 数据结构图
    dot_structure = create_data_structure_diagram()
    dot_structure.render('/workspace/data_structure', view=False, cleanup=True)
    print("✓ 数据结构图已生成: data_structure.png")
    
    print("\n所有流程图生成完成！")
    print("- pipeline_detailed.png: 完整的详细流程图")
    print("- pipeline_simplified.png: 简化的高层流程图")
    print("- data_structure.png: 数据文件组织结构图")
