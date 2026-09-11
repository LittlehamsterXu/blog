---
title: "安全 Security"
description: "Reverse Engineering、Pwn、CTF、漏洞分析与系统安全技术积累。"
summary: "传统安全基础是当前 Compiler Security 研究的起点，而不是需要隐藏的旧内容。"
ShowToc: true
---

## Security Foundations / Earlier Work

我过去主要通过 CTF 中的 Reverse Engineering 和 Pwn 训练，建立了二进制程序分析、漏洞利用、调试和系统行为理解的基础。后续研究转向编译器，并不意味着这些内容失效；相反，它们帮助我理解编译器输出、运行时行为、内存安全问题和攻击面。

## Reverse Engineering

关注程序结构恢复、静态与动态分析、汇编阅读、调用约定、编译器生成代码和运行时行为。相关内容适合记录具体样本、分析过程和工具使用，而不是只保留最终结论。

## Binary Exploitation / Pwn

保留栈、堆、内存破坏、ROP、格式化字符串、沙箱和 mitigations 等技术积累。之后会优先整理具有通用分析价值的内容，并在文章中明确实验环境和复现条件。

## CTF Writeups

CTF writeup 作为问题求解和安全基础训练的记录保留。内容可以按题目类型、漏洞原语和分析方法整理，不把它们从网站中清除或简单归档为“过时内容”。

## Vulnerability Research / Systems Security

这一部分与当前 Compiler Security 方向相连：编译器 bug 可能表现为错误代码生成、越界访问、内存安全问题或难以观察的语义偏差。安全背景页会持续补充漏洞分析、系统机制和测试方法之间的联系。

## Content Organization

后续安全文章优先使用稳定标签：`Reverse Engineering`、`Pwn`、`Binary Security`、`CTF`、`Vulnerability Research`、`Systems Security`。不会为每一篇文章创建一次性标签。
