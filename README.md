# 资产定价与机器学习：从理论到 A 股实证

本地阅读的自包含教材，配套模拟市场生成器 `apsim`。

## 安装

```bash
uv sync
```

## 构建与阅读

```bash
sh scripts/build.sh   # 构建全书（执行全部代码）
sh scripts/serve.sh   # 启动本地服务器，浏览器打开 http://localhost:8000
```

写作时可用实时预览：`cd book && uv run jupyter book start --execute`。

## 测试

```bash
uv run pytest
```

## 目录

- `book/`：书的源文件（MyST Markdown）
- `src/apsim/`：模拟市场生成器与工具函数
- `tests/`：单元测试
- `docs/specs/`：设计文档
- `参考资料/`：参考材料
