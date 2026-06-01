#!/usr/bin/env python
# -*- coding: utf-8 -*-

def reformat_code(code):
    lines = code.split('\n')
    result_lines = []
    
    for line in lines:
        stripped = line.lstrip().rstrip()
        if stripped:
            result_lines.append(stripped)
    
    return '\n'.join(result_lines) + '\n'

def main():
    input_file = '/root/autodl-tmp/dit/1Prompt1Story/resource/source_code.txt'
    backup_file = '/root/autodl-tmp/dit/1Prompt1Story/resource/source_code_no_comments.txt'
    
    print(f"从备份文件读取: {backup_file}")
    with open(backup_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    print("重新排版（删除空行和缩进，每行代码结束后换行）...")
    formatted_code = reformat_code(code)
    
    print(f"保存到: {input_file}")
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(formatted_code)
    
    original_lines = len([l for l in code.split('\n') if l.strip()])
    new_lines = len(formatted_code.split('\n'))
    original_size = len(code)
    new_size = len(formatted_code)
    
    print(f"完成！")
    print(f"原始有效行数: {original_lines}, {original_size} 字符")
    print(f"处理后: {new_lines} 行, {new_size} 字符")
    print(f"减少: {original_size - new_size} 字符")

if __name__ == "__main__":
    main()
