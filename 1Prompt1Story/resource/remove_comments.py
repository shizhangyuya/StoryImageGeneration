#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import io

def remove_comments(code):
    result = []
    lines = code.split('\n')
    in_multiline = False
    multiline_char = None
    
    for line in lines:
        if in_multiline:
            end_idx = line.find(multiline_char * 3)
            if end_idx != -1:
                in_multiline = False
                line = line[end_idx + 3:].lstrip()
                if not line.strip():
                    continue
            else:
                continue
        
        if not in_multiline:
            if '"""' in line:
                parts = line.split('"""')
                if len(parts) >= 3:
                    line = parts[0] + parts[-1] if len(parts) % 2 == 1 else parts[0]
                elif len(parts) == 2:
                    in_multiline = True
                    multiline_char = '"'
                    line = parts[0].rstrip()
                else:
                    idx = line.find('"""')
                    line = line[:idx].rstrip()
            elif "'''" in line:
                parts = line.split("'''")
                if len(parts) >= 3:
                    line = parts[0] + parts[-1] if len(parts) % 2 == 1 else parts[0]
                elif len(parts) == 2:
                    in_multiline = True
                    multiline_char = "'"
                    line = parts[0].rstrip()
                else:
                    idx = line.find("'''")
                    line = line[:idx].rstrip()
            
            if not in_multiline and line.strip():
                in_string = False
                string_char = None
                comment_pos = -1
                
                i = 0
                while i < len(line):
                    if line[i] in ['"', "'"]:
                        if not in_string:
                            in_string = True
                            string_char = line[i]
                            if i + 2 < len(line) and line[i:i+3] == string_char * 3:
                                i += 2
                        elif line[i] == string_char:
                            if i + 2 < len(line) and line[i:i+3] == string_char * 3:
                                i += 2
                            else:
                                in_string = False
                                string_char = None
                    elif line[i] == '#' and not in_string:
                        comment_pos = i
                        break
                    i += 1
                
                if comment_pos >= 0:
                    line = line[:comment_pos].rstrip()
        
        if line.strip() or (not in_multiline and not line.strip()):
            result.append(line)
    
    return '\n'.join(result)

def main():
    input_file = '/root/autodl-tmp/dit/1Prompt1Story/resource/source_code.txt'
    output_file = '/root/autodl-tmp/dit/1Prompt1Story/resource/source_code_no_comments.txt'
    
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    print("去除注释...")
    code_no_comments = remove_comments(code)
    
    print(f"保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code_no_comments)
    
    original_lines = len(code.split('\n'))
    new_lines = len(code_no_comments.split('\n'))
    print(f"完成！原始行数: {original_lines}, 处理后行数: {new_lines}")

if __name__ == "__main__":
    main()
