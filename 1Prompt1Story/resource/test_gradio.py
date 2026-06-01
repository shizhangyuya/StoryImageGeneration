#!/usr/bin/env python
"""
简单的 Gradio 测试脚本，用于诊断问题
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import traceback

def test_function(text):
    """测试函数"""
    try:
        return f"输入的内容: {text}"
    except Exception as e:
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def test_interface():
    """创建测试界面"""
    with gr.Blocks(title="测试界面") as demo:
        gr.Markdown("# 测试界面")
        
        with gr.Row():
            input_text = gr.Textbox(label="输入文本", value="测试")
            output_text = gr.Textbox(label="输出文本")
        
        test_button = gr.Button("测试", variant="primary")
        
        test_button.click(
            fn=test_function,
            inputs=[input_text],
            outputs=[output_text]
        )
    
    return demo

if __name__ == "__main__":
    print("启动测试界面...")
    demo = test_interface()
    try:
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7861,
            show_error=True,
            enable_queue=True
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print(traceback.format_exc())
        # 尝试使用 share=True
        print("\n尝试使用公共链接模式...")
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7861,
            show_error=True,
            enable_queue=True
        )

