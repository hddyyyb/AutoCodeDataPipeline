# 只展示【新增 / 修改】的关键部分，其余你文件保持不变
# ↓↓↓ 请把下面函数插入到文件中（在 _generate_once 后）↓↓↓

def rewrite_json_to_template(model, tokenizer, bad_output: str, prompt: str, lang: str) -> str:
    if lang == "zh":
        extra = """
【改写任务】
你刚才的输出是JSON/类JSON。
请严格改写为【纯文本模板】，必须包含以下标题：
【结论】【关键判断与处理分支】【证据引用】【推理过程】
禁止输出JSON，禁止新增标题。
【证据引用】只能使用白名单里的文件与chunk_id。
【推理过程】每一步必须以(chunk_id=16位十六进制)结尾。
只输出改写后的正文。
"""
    else:
        extra = """
[REWRITE TASK]
Rewrite the previous JSON-like output into the required plain-text template.
DO NOT output JSON.
Use only whitelist references.
"""

    rewrite_prompt = (
        prompt
        + "\n\n"
        + extra
        + "\n\n[JSON OUTPUT]\n"
        + bad_output
        + "\n\n### Response\n"
    )
    return _generate_once(model, tokenizer, rewrite_prompt)
