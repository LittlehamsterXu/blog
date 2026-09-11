---
title: "开源贡献 Contributions"
description: "整理 Apache TVM、LLVM / MLIR、IREE、Triton 等项目中的技术问题和 upstream 结果。"
summary: "关注有技术含量、有 upstream 反馈和可复用经验的问题，而不是 GitHub activity feed。"
ShowToc: true
---

这里整理值得长期维护的 issue、PR、修复和 upstream 反馈。状态以链接页面为准；后续新增记录时尽量补充 Problem、Status、Upstream response 和最终 Fix。

## Apache TVM

| Issue / PR | Problem | Status | Upstream result |
| --- | --- | --- | --- |
| [#20125](https://github.com/apache/tvm/issues/20125) | LLVM backend allocation extent truncation | Fixed | 相关第三方修复已合入上游 |
| [#20315](https://github.com/apache/tvm/issues/20315) | FP8 E5M2 zero / subnormal conversion correctness | Open / triage | 持续跟踪 issue 讨论 |
| [#20273](https://github.com/apache/tvm/issues/20273) · [PR #20310](https://github.com/apache/tvm/pull/20310) | CUDA sub-byte shared-memory allocation | Related PR | 跟踪相关实现和 review 结果 |

## LLVM / MLIR、IREE、Triton

这些项目是后续重点整理对象。新增条目时不追求数量，优先记录能够说明根因、测试缺口、修复设计或 compiler semantics 的代表性问题。

## Contribution Record Format

每条记录建议包含：

- **Project**：项目和组件
- **Issue / PR**：可直接访问的 upstream 链接
- **Problem**：最小、准确的问题描述
- **Status**：open、triage、fixed、merged 或 superseded
- **Upstream response**：维护者反馈、review 或讨论结论
- **Fix / merged PR**：最终修复和关联提交
- **Lesson**：对测试、工具链或安全研究的可复用经验

这类记录比简单展示提交次数更能体现工程判断和研究价值。
